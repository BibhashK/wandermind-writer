"""
FastAPI backend for Wandermind Writer.

WHAT A BACKEND IS:
    A URL comes in → a Python function runs → data goes back.
    FastAPI wires URLs to functions with @app.get / @app.post decorators.

WHY WE STREAM:
    Generating takes 20-50 seconds. Rather than make the user stare at a spinner,
    the server pushes an update as each step finishes and the UI renders it live.

SECURITY:
    Credentials arrive with each request, live in memory for that request only,
    and are never written to disk, logged, or persisted. We log events, never
    payloads.
"""

import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from . import agent
from .agent import AgentError
from .models import GenerateRequest, ScoutRequest, PublishRequest, UserConfig
from .seo import score_article

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="Wandermind Writer",
    description="AI research-to-publish agent for WordPress. Bring your own keys.",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # TODO: restrict to your domain before going live
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
#  HEALTH
# =============================================================================

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "wandermind-writer"}


# =============================================================================
#  VERIFY
# =============================================================================

@app.post("/api/verify")
def verify(config: UserConfig):
    """
    Check the credentials before running anything expensive.

    Failing here takes two seconds. Failing at the last step of a 40-second run
    wastes the user's time and their quota.
    """
    try:
        wp = agent.verify_wordpress(
            config.clean_wp_url(), config.wp_username, config.wp_app_password
        )
        # A one-word call proves the Gemini key works without burning real quota.
        agent.call_llm("Reply with one word: ok", config.google_api_key, mode="fast")
        return {"ok": True, "wordpress_user": wp["name"]}
    except AgentError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
#  SCOUT
# =============================================================================

@app.post("/api/scout")
def scout(req: ScoutRequest):
    """Five ranked article ideas, tuned to whatever the user's blog covers."""
    try:
        topics = agent.scout_topics(
            req.niche,
            req.config.tavily_api_key,
            req.config.google_api_key,
            req.mode,
        )
        return {"topics": topics}
    except AgentError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
#  GENERATE — streamed
# =============================================================================

def sse(event: str, data: dict) -> str:
    """One Server-Sent Event. The wire format is literally: data: {...}\\n\\n"""
    return f"data: {json.dumps({'event': event, **data})}\n\n"


def generation_stream(req: GenerateRequest):
    """
    Run the pipeline, yielding a progress event after each step.

    This is a Python generator — each `yield` sends a chunk to the browser
    immediately rather than waiting for the whole function to finish.
    """
    cfg = req.config
    wp_url = cfg.clean_wp_url()
    mode = req.mode

    try:
        # --- SEO + RESEARCH, CONCURRENTLY ------------------------------------
        # These don't depend on each other, so there's no reason to wait.
        yield sse("progress", {"step": "seo", "message": "Optimising for search…"})

        with ThreadPoolExecutor(max_workers=2) as pool:
            seo_future = pool.submit(
                agent.generate_seo, req.topic, cfg.google_api_key, mode
            )
            research_future = pool.submit(
                agent.research_topic, req.topic, cfg.tavily_api_key
            )

            seo_data = seo_future.result()
            yield sse("seo", {
                "seo_title": seo_data["seo_title"],
                "keyword": seo_data["keyword"],
                "meta_description": seo_data["meta_description"],
            })

            yield sse("progress", {"step": "research", "message": "Reading the live web…"})
            research_data = research_future.result()

        yield sse("research", {"sources": research_data["sources"]})

        # --- INTERNAL LINKS ---------------------------------------------------
        yield sse("progress", {"step": "links", "message": "Finding your posts to link…"})
        auth = agent.wp_auth(cfg.wp_username, cfg.wp_app_password)
        internal_links = agent.get_internal_links(wp_url, auth)
        yield sse("links", {"count": len(internal_links)})

        # --- WRITE ------------------------------------------------------------
        yield sse("progress", {"step": "write", "message": "Writing the article…"})
        draft = agent.write_article(
            seo_data["seo_title"],
            seo_data["keyword"],
            research_data["research"],
            research_data["sources"],
            internal_links,
            cfg.google_api_key,
            mode,
        )
        title, body = agent.split_title_body(draft)
        yield sse("article", {"title": title, "body": body})

        # --- SCORE (pure Python — instant) ------------------------------------
        yield sse("progress", {"step": "score", "message": "Scoring on-page SEO…"})
        yield sse("score", score_article(
            title, body, seo_data["keyword"], seo_data["meta_description"]
        ))

        # --- LINKEDIN ---------------------------------------------------------
        yield sse("progress", {"step": "linkedin", "message": "Writing the LinkedIn post…"})
        linkedin = agent.write_linkedin_post(
            seo_data["seo_title"], draft, cfg.google_api_key, mode
        )
        yield sse("linkedin", {"post": linkedin})

        yield sse("done", {"message": "Ready."})

    except AgentError as e:
        log.warning("generation failed: %s", e)
        yield sse("error", {"message": str(e)})
    except Exception as e:
        log.exception("unexpected error during generation")
        yield sse("error", {"message": f"Something broke: {e}"})


@app.post("/api/generate")
def generate(req: GenerateRequest):
    return StreamingResponse(
        generation_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",     # stops nginx buffering the stream
        },
    )


# =============================================================================
#  PUBLISH
# =============================================================================

@app.post("/api/publish")
def publish(req: PublishRequest):
    """
    Push to WordPress as a draft.

    Deliberately separate from /generate: the user reads and edits first. The
    agent never publishes anything unreviewed.
    """
    try:
        return agent.publish_draft(
            req.config.clean_wp_url(),
            req.config.wp_username,
            req.config.wp_app_password,
            req.title,
            req.body_markdown,
            req.meta_description,
            req.category,
        )
    except AgentError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
#  FRONTEND
# =============================================================================
# One server serves both the API and the static files, so there's one thing to
# deploy rather than two.

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/guide")
    def guide():
        return FileResponse(FRONTEND_DIR / "guide.html")