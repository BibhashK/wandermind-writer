"""
The LangGraph agent, refactored for multi-user use.

KEY CHANGE FROM THE PERSONAL VERSION:
    Config is PASSED IN, not read from a .env file. Every function that needs a
    credential receives it as an argument. This is what makes the agent
    multi-user: two people can call it simultaneously with different keys, and
    neither request can see the other's config.

    (In engineering terms this is "dependency injection". It also makes the code
    testable — you can pass fake keys in a test without touching the environment.)

SECURITY:
    Credentials exist only in memory, only for the life of one request. They are
    never written to disk, never logged, never persisted anywhere.
"""

import json
import time
import logging

import requests
from requests.auth import HTTPBasicAuth
import markdown
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI

log = logging.getLogger(__name__)

# Models are tried in this order. Each has its own free-tier quota, so falling
# back across them routes around a single overloaded or exhausted model.
MODEL_FALLBACK_CHAIN = [
    "gemma-4-31b-it",
    "gemini-2.0-flash",
    "gemini-flash-latest",
]

RETRYABLE_ERRORS = ("503", "UNAVAILABLE", "overloaded", "RESOURCE_EXHAUSTED", "429")


class AgentError(Exception):
    """Raised when the agent fails in a way the user should see."""


# =============================================================================
#  LOW-LEVEL CLIENTS
# =============================================================================

def call_llm(prompt: str, api_key: str, temperature: float = 0.7) -> str:
    """
    Send a prompt to the LLM and return plain text.

    Retries on transient errors, then falls back to the next model in the chain.
    The API key is passed in by the caller — never read from the environment.
    """
    last_error: Exception | None = None

    for model_name in MODEL_FALLBACK_CHAIN:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
        )

        for attempt in range(3):
            try:
                response = llm.invoke(prompt)
                content = response.content

                # Some models return a list of content parts, not a plain string.
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                return content

            except Exception as e:
                last_error = e
                if any(marker in str(e) for marker in RETRYABLE_ERRORS):
                    wait = (attempt + 1) * 4
                    log.warning("%s busy (attempt %d). Waiting %ds", model_name, attempt + 1, wait)
                    time.sleep(wait)
                else:
                    # A real error (bad key, malformed request) — don't mask it.
                    raise AgentError(f"Model error: {e}") from e

    raise AgentError(
        "All models are busy or your quota is exhausted. Try again in a few minutes. "
        f"(last error: {last_error})"
    )


def parse_json_response(raw: str):
    """
    Parse JSON from an LLM response.

    Models often wrap JSON in ```json fences despite being told not to, so we
    strip those before parsing. Raises AgentError with the raw text if it still
    fails, so the user sees what went wrong rather than a bare stack trace.
    """
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise AgentError(f"The model returned malformed JSON: {cleaned[:200]}") from e


def tavily_search(query: str, api_key: str, max_results: int = 5) -> list[dict]:
    """Run one Tavily search. Returns [] on failure rather than crashing the run."""
    try:
        client = TavilyClient(api_key=api_key)
        results = client.search(
            query=query,
            max_results=max_results,
            topic="news",
            search_depth="advanced",
        )
        return results.get("results", [])
    except Exception as e:
        log.warning("Search failed for '%s': %s", query, e)
        return []


# =============================================================================
#  WORDPRESS
# =============================================================================

def wp_auth(username: str, app_password: str) -> HTTPBasicAuth:
    return HTTPBasicAuth(username, app_password)


def verify_wordpress(wp_url: str, username: str, app_password: str) -> dict:
    """
    Check the WordPress credentials work before we do expensive AI calls.

    Failing fast here saves the user from watching a 60-second generation run
    only to hit a 401 at the very end.
    """
    try:
        r = requests.get(
            f"{wp_url}/wp-json/wp/v2/users/me",
            auth=wp_auth(username, app_password),
            timeout=15,
        )
    except requests.RequestException as e:
        raise AgentError(f"Couldn't reach {wp_url}: {e}") from e

    if r.status_code == 200:
        data = r.json()
        return {"ok": True, "name": data.get("name", username)}

    if r.status_code in (401, 403):
        raise AgentError(
            "WordPress rejected those credentials. Check the username and "
            "application password. If you use a security plugin (e.g. Wordfence), "
            "it may be blocking REST API authentication."
        )

    raise AgentError(f"WordPress returned {r.status_code}: {r.text[:200]}")


def get_category_id(name: str, wp_url: str, auth) -> int | None:
    """Resolve a category NAME to its numeric ID. Returns None if not found."""
    try:
        r = requests.get(
            f"{wp_url}/wp-json/wp/v2/categories",
            params={"search": name},
            auth=auth,
            timeout=15,
        )
        for cat in r.json():
            if cat["name"].lower() == name.lower():
                return cat["id"]
    except Exception as e:
        log.warning("Category lookup failed: %s", e)
    return None


def get_internal_links(wp_url: str, auth, limit: int = 20) -> list[dict]:
    """
    Fetch the user's published posts so the writer can link to them.

    Internal linking is one of the few SEO levers you fully control — it keeps
    readers on the site and spreads page authority. Drafts are excluded because
    they have no public URL.
    """
    try:
        r = requests.get(
            f"{wp_url}/wp-json/wp/v2/posts",
            params={"per_page": limit, "status": "publish", "_fields": "title,link"},
            auth=auth,
            timeout=20,
        )
        if r.status_code != 200:
            return []
        return [
            {"title": p["title"]["rendered"].strip(), "url": p["link"]}
            for p in r.json()
            if p.get("link") and p.get("title", {}).get("rendered")
        ]
    except Exception as e:
        log.warning("Couldn't fetch internal links: %s", e)
        return []


def publish_draft(
    wp_url: str,
    username: str,
    app_password: str,
    title: str,
    body_markdown: str,
    meta_description: str,
    category: str,
) -> dict:
    """
    Push an article to WordPress as a DRAFT. Never publishes automatically —
    the human always reviews before anything goes live.
    """
    auth = wp_auth(username, app_password)

    # markdown → HTML. This also turns [text](url) into real <a> anchors, which
    # is what makes the internal/external linking actually work once published.
    body_html = markdown.markdown(body_markdown, extensions=["extra"])

    payload = {
        "title": title,
        "content": body_html,
        "status": "draft",
        "excerpt": meta_description,
    }

    category_id = get_category_id(category, wp_url, auth)
    if category_id is not None:
        payload["categories"] = [category_id]

    try:
        r = requests.post(
            f"{wp_url}/wp-json/wp/v2/posts", json=payload, auth=auth, timeout=30
        )
    except requests.RequestException as e:
        raise AgentError(f"Couldn't reach WordPress: {e}") from e

    if r.status_code == 201:
        data = r.json()
        post_id = data.get("id")
        return {
            "post_id": post_id,
            "edit_url": f"{wp_url}/wp-admin/post.php?post={post_id}&action=edit",
            "category_matched": category_id is not None,
        }

    raise AgentError(f"WordPress error {r.status_code}: {r.text[:250]}")


# =============================================================================
#  AGENT STEPS
# =============================================================================

def scout_topics(tavily_key: str, google_key: str) -> list[dict]:
    """Scan the news across several angles and rank the five best article ideas."""
    angles = [
        "biggest tech news this week",
        "AI breakthrough trending now",
        "new developer tools launch",
        "Europe technology policy news",
        "enterprise AI adoption news",
    ]

    results = []
    for angle in angles:
        results.extend(tavily_search(angle, tavily_key, max_results=4))

    if not results:
        raise AgentError("Couldn't fetch any news. Check your Tavily API key.")

    scan = "\n\n".join(f"- {r['title']}: {r['content'][:400]}" for r in results)

    prompt = f"""You are a ruthless editor at a major tech publication. Below is a raw
scan of current tech/AI news. Identify the FIVE strongest blog post ideas for an
independent tech blogger covering AI, agents, developer tools, and enterprise
technology, with a European perspective.

Judge each on:
- FRESHNESS: genuinely new, or already everywhere?
- TRAFFIC POTENTIAL: would people search for or click this?
- CROWDEDNESS: is every outlet already covering it? (less crowded is better)
- HOOK STRENGTH: is there a surprising number, angle, or tension to open with?
- AUTHORITY FIT: can an enterprise/AI engineer credibly own this take?

Reject generic ideas like "The Future of AI". Favour specific, timely angles with
a real news peg.

Return ONLY valid JSON — exactly 5 objects, no markdown fences:
[
  {{
    "topic": "the specific topic, as a clear phrase",
    "why": "one sentence on why this could pull traffic",
    "hook": "the surprising fact or tension to open with",
    "category": "AI or Tech or Business"
  }}
]

NEWS SCAN:
{scan[:6000]}
"""
    return parse_json_response(call_llm(prompt, google_key))


def generate_seo(topic: str, google_key: str) -> dict:
    """Turn a topic into an SEO title, primary keyword, and meta description."""
    prompt = f"""You are an SEO strategist for a tech blog about AI and software.

TOPIC: {topic}

Return ONLY valid JSON, no markdown fences:
{{
  "seo_title": "compelling, search-friendly title, 30-60 characters",
  "keyword": "the single primary keyword phrase to target",
  "meta_description": "120-160 character meta description containing the keyword"
}}"""
    return parse_json_response(call_llm(prompt, google_key))


def research_topic(topic: str, tavily_key: str) -> dict:
    """Deep-dive the chosen topic. Keeps source URLs so the writer can cite them."""
    results = tavily_search(topic, tavily_key, max_results=6)

    if not results:
        raise AgentError("No research results found. Try a different topic.")

    sources = [
        {"title": r["title"], "url": r["url"], "snippet": r["content"][:300]}
        for r in results
        if r.get("url")
    ]

    research_text = "\n\n".join(
        f"- {r['title']} (source: {r.get('url', 'n/a')}): {r['content']}"
        for r in results
    )

    return {"research": research_text, "sources": sources}


def write_article(
    seo_title: str,
    keyword: str,
    research: str,
    sources: list[dict],
    internal_links: list[dict],
    google_key: str,
) -> str:
    """
    Write the article with real editorial craft and inline SEO links.

    The prompt is the product here. It bans the tells of AI writing and demands
    what good journalism actually does: a concrete lede, real tension, specific
    names and numbers, a point of view, and an earned close.
    """
    external = "\n".join(f"- {s['title']} → {s['url']}" for s in sources) or "(none)"
    internal = (
        "\n".join(f"- {p['title']} → {p['url']}" for p in internal_links)
        or "(none available — skip internal links)"
    )

    prompt = f"""You are a senior features writer for a publication like The Wall Street
Journal or Forbes. Your readers are intelligent and busy — engineers, founders,
enterprise leaders. They stop reading the instant you become generic.

TITLE (first line, no markdown symbols): {seo_title}
PRIMARY KEYWORD (weave in naturally, never stuff): {keyword}

=== CRAFT RULES ===

1. THE LEDE. Open with something CONCRETE and SPECIFIC from the research — a
   startling number, a named person doing a specific thing, a sharp contrast.
   FORBIDDEN: "In today's rapidly evolving...", "In the age of AI...",
   "Imagine a world where...", "Technology is changing fast..."

2. TENSION. Every piece worth reading has a conflict: old way vs new way,
   promised vs shipped, who wins vs who loses. State it early; let it drive.

3. SPECIFICITY. Use the actual names, companies, numbers, and dates from the
   research. "Adoption is growing" is worthless.

4. A POINT OF VIEW. Take a position. Acknowledge the counter-argument, then say
   what you think anyway.

5. RHYTHM. Vary sentence length. Short sentences land hard. Then a longer one
   that develops the idea and gives the reader room to breathe.

6. NO AI TELLS. Banned: "delve", "landscape", "realm", "testament to", "it's
   important to note", "in conclusion", "game-changer", "revolutionize",
   "unlock the potential", "navigate the complexities". Never end a section by
   summarising what you just said.

7. THE CLOSE. End with a sharp, earned conclusion — an implication, a prediction
   with a reason, or a question that lingers. Never "Only time will tell."

=== LINKING (matters for SEO) ===

Use markdown links: [descriptive anchor](url)

EXTERNAL — cite sources inline, on the SPECIFIC claim each supports.
  • 3-5 links. Descriptive anchor text, never "click here" or a bare URL.
  • ONLY use URLs from this list. NEVER invent a URL.
{external}

INTERNAL — link to the author's own earlier posts where genuinely relevant.
  • 1-3 links, woven into sentences. Never a "Related posts" dump.
  • If none are truly relevant, use none. Forced links hurt.
  • ONLY use URLs from this list. NEVER invent a URL.
{internal}

=== FORMAT ===
- Title on the very first line.
- 700-900 words.
- Use ## for subheadings. Make them interesting, not labels.
  (Good: "The quiet cost nobody priced in". Bad: "Challenges and Considerations")
- Every factual claim must come from the research. Invent nothing.

RESEARCH:
{research}
"""
    return call_llm(prompt, google_key).strip()


def write_linkedin_post(seo_title: str, article: str, google_key: str) -> str:
    """Write a LinkedIn hook post that funnels readers to the article."""
    prompt = f"""You are a LinkedIn content strategist. Write a post promoting the
article below. Goal: stop the scroll, deliver real value, drive clicks, win followers.

STRUCTURE:
1. HOOK — 1-2 short lines. LinkedIn truncates after ~2 lines, so this must earn
   the "see more". Use the sharpest number or claim in the piece.
2. VALUE — 3-4 punchy insights, each on its own line, blank line between. The
   reader should learn something even if they never click.
3. CURIOSITY GAP — one line teasing what else is in the full article.
4. CTA — a soft invitation to read, and to follow for more.
5. HASHTAGS — 4-5 relevant, on the last line.

TONE: professional and insightful. Confident, not hypey. Max 2 emojis.
Short lines, plenty of white space. First person, as the author.
No "In today's world" openers. No corporate filler.

ARTICLE TITLE: {seo_title}
ARTICLE:
{article[:3000]}
"""
    return call_llm(prompt, google_key).strip()


def split_title_body(draft: str) -> tuple[str, str]:
    """First line is the title; everything after is the body."""
    lines = draft.strip().split("\n", 1)
    title = lines[0].lstrip("# ").strip().strip("*").strip()
    body = lines[1].strip() if len(lines) > 1 else ""
    return title, body