"""
FastAPI backend for Wandermind Writer.

WHAT A BACKEND IS:
    A URL comes in → a Python function runs → data goes back out.
    FastAPI wires URLs to functions using @app.get / @app.post decorators.

WHY WE STREAM:
    Generating an article takes 30-60 seconds. If we just made the user stare at
    a spinner, it would feel broken. Instead we use Server-Sent Events (SSE): the
    server pushes progress updates as each step finishes, and the UI shows them
    live. This is the difference between "is it stuck?" and "ooh, it's working."

SECURITY (the BYOK promise):
    User credentials arrive with each request, live in memory for that request
    only, and are never written to disk, never logged, and never persisted.
    We log events, never payloads.
"""

import json
import logging
from pathlib import Path

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
    version="0.1.0",
)

# CORS lets the browser talk to this API. In development the frontend may be
# served from a different port, so we allow it. Tighten this before going live.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # TODO: restrict to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
#  HEALTH
# =============================================================================

@app.get("/api/health")
def health():
    """Simple liveness check. Useful for uptime monitoring and deploy checks."""
    return {"status": "ok", "service": "wandermind-writer"}


# =============================================================================
#  VERIFY CREDENTIALS
# =============================================================================

@app.post("/api/verify")
def verify(config: UserConfig):
    """
    Check the user's credentials BEFORE running an expensive generation.

    Failing fast here is a real UX win: nobody wants to watch a 60-second run
    only to hit a 401 at the very last step.
    """
    try:
        wp = agent.verify_wordpress(
            config.clean_wp_url(), config.wp_username, config.wp_app_password
        )
        # A tiny LLM call proves the Gemini key works without burning real quota.
        agent.call_llm("Reply with the single word: ok", config.google_api_key)

        return {"ok": True, "wordpress_user": wp["name"]}

    except AgentError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
#  SCOUT — find trending topics
# =============================================================================

@app.post("/api/scout")
def scout(req: ScoutRequest):
    """Scan the news and return five ranked article ideas."""
    try:
        topics = agent.scout_topics(
            req.config.tavily_api_key, req.config.google_api_key
        )
        return {"topics": topics}
    except AgentError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
#  GENERATE — the main event, streamed
# =============================================================================

def sse(event: str, data: dict) -> str:
    """
    Format one Server-Sent Event.

    The wire format is literally:  data: {...json...}\n\n
    The browser's EventSource / fetch-reader picks these up as they arrive.
    """
    return f"data: {json.dumps({'event': event, **data})}\n\n"


def generation_stream(req: GenerateRequest):
    """
    Run the full pipeline, yielding a progress event after each step.

    This is a Python generator: each `yield` sends a chunk to the browser
    immediately rather than waiting for the whole function to finish. That's
    what makes the progress feel live.
    """
    cfg = req.config
    wp_url = cfg.clean_wp_url()

    try:
        # --- 1. SEO metadata ---------------------------------------------------
        yield sse("progress", {"step": "seo", "message": "Optimising for search…"})
        seo_data = agent.generate_seo(req.topic, cfg.google_api_key)
        yield sse("seo", {
            "seo_title": seo_data["seo_title"],
            "keyword": seo_data["keyword"],
            "meta_description": seo_data["meta_description"],
        })

        # --- 2. Research -------------------------------------------------------
        yield sse("progress", {"step": "research", "message": "Researching the live web…"})
        research_data = agent.research_topic(req.topic, cfg.tavily_api_key)
        yield sse("research", {"sources": research_data["sources"]})

        # --- 3. Internal links -------------------------------------------------
        yield sse("progress", {"step": "links", "message": "Finding your posts to link…"})
        auth = agent.wp_auth(cfg.wp_username, cfg.wp_app_password)
        internal_links = agent.get_internal_links(wp_url, auth)
        yield sse("links", {"count": len(internal_links)})

        # --- 4. Write ----------------------------------------------------------
        yield sse("progress", {"step": "write", "message": "Writing the article…"})
        draft = agent.write_article(
            seo_data["seo_title"],
            seo_data["keyword"],
            research_data["research"],
            research_data["sources"],
            internal_links,
            cfg.google_api_key,
        )
        title, body = agent.split_title_body(draft)
        yield sse("article", {"title": title, "body": body})

        # --- 5. SEO score ------------------------------------------------------
        yield sse("progress", {"step": "score", "message": "Scoring on-page SEO…"})
        score = score_article(
            title, body, seo_data["keyword"], seo_data["meta_description"]
        )
        yield sse("score", score)

        # --- 6. LinkedIn post --------------------------------------------------
        yield sse("progress", {"step": "linkedin", "message": "Writing the LinkedIn post…"})
        linkedin = agent.write_linkedin_post(
            seo_data["seo_title"], draft, cfg.google_api_key
        )
        yield sse("linkedin", {"post": linkedin})

        # --- Done --------------------------------------------------------------
        yield sse("done", {"message": "Ready. Review it, then push to WordPress."})

    except AgentError as e:
        log.warning("Generation failed: %s", e)
        yield sse("error", {"message": str(e)})
    except Exception as e:
        log.exception("Unexpected error during generation")
        yield sse("error", {"message": f"Unexpected error: {e}"})


@app.post("/api/generate")
def generate(req: GenerateRequest):
    """Generate an article + LinkedIn post, streaming progress as it goes."""
    return StreamingResponse(
        generation_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # stops nginx buffering the stream
        },
    )


# =============================================================================
#  PUBLISH — push the (possibly user-edited) article to WordPress
# =============================================================================

@app.post("/api/publish")
def publish(req: PublishRequest):
    """
    Push the article to WordPress as a DRAFT.

    Deliberately a SEPARATE endpoint from /generate. The user gets to read and
    edit the article first — the agent never publishes anything unreviewed.
    """
    try:
        result = agent.publish_draft(
            req.config.clean_wp_url(),
            req.config.wp_username,
            req.config.wp_app_password,
            req.title,
            req.body_markdown,
            req.meta_description,
            req.category,
        )
        return result
    except AgentError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
#  SERVE THE FRONTEND
# =============================================================================
# In production the same server serves both the API and the static frontend, so
# there's only one thing to deploy.

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")