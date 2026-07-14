"""
The agent. Config is passed in, so it serves many users at once.

SPEED:
    1. MODEL TIERS. "Fast" leads with Flash-Lite (~381 tok/s, Pro-derived).
       "Quality" leads with 3.5 Flash (better factual grounding). Both fall back
       to a shared stable model so a busy tier never kills the run.
    2. PARALLELISM. Searches fire at once instead of one after another, and SEO
       runs while research is still going. Sequential code waits; concurrent
       code doesn't.

SECURITY:
    Credentials live in memory for one request. Never written to disk, never
    logged, never persisted.
"""

import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.auth import HTTPBasicAuth
import markdown
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI

log = logging.getLogger(__name__)


# =============================================================================
#  MODEL TIERS
# =============================================================================
# NOTE: gemini-2.0-flash and 2.0-flash-lite were SHUT DOWN on 1 June 2026.
# Don't put them back — requests to them 404.

MODEL_CHAINS = {
    # Flash-Lite is the fastest tier and is derived from Pro rather than a
    # smaller Flash base — quick without being dumb.
    "fast": [
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
    ],
    # 3.5 Flash beats 3.1 Pro on agentic benchmarks while running faster, and
    # grounds facts better than Flash-Lite — which matters when citing sources.
    "quality": [
        "gemini-3.5-flash",
        "gemini-3-flash",
        "gemini-2.5-flash",
    ],
}

RETRYABLE = (
    "503", "500", "502", "504",
    "UNAVAILABLE", "INTERNAL", "overloaded",
    "RESOURCE_EXHAUSTED", "429",
)


class AgentError(Exception):
    """A failure the user should see, phrased for a human."""


# =============================================================================
#  LLM
# =============================================================================

def call_llm(prompt: str, api_key: str, mode: str = "fast", temperature: float = 0.7) -> str:
    """Send a prompt to the LLM. Retries transient errors, then falls back a tier."""
    chain = MODEL_CHAINS.get(mode, MODEL_CHAINS["fast"])
    last_error: Exception | None = None

    for model_name in chain:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
        )

        for attempt in range(2):
            try:
                response = llm.invoke(prompt)
                content = response.content

                if isinstance(content, list):
                    content = "".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    )
                log.info("used %s (%s mode)", model_name, mode)
                return content

            except Exception as e:
                last_error = e
                if any(m in str(e) for m in RETRYABLE):
                    time.sleep(3 * (attempt + 1))
                else:
                    raise AgentError(f"Model error: {e}") from e

    raise AgentError(
        "Every model is busy or your quota is used up. Wait a few minutes and "
        f"try again. (last error: {last_error})"
    )


def parse_json_response(raw: str):
    """Parse JSON from an LLM reply, stripping the ```json fences they add anyway."""
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise AgentError(f"The model returned malformed JSON: {cleaned[:200]}") from e


# =============================================================================
#  SEARCH — parallel
# =============================================================================

def tavily_search(query: str, api_key: str, max_results: int = 5) -> list[dict]:
    """One search. Returns [] on failure rather than killing the run."""
    try:
        client = TavilyClient(api_key=api_key)
        return client.search(
            query=query,
            max_results=max_results,
            topic="news",
            search_depth="advanced",
        ).get("results", [])
    except Exception as e:
        log.warning("search failed for %r: %s", query, e)
        return []


def parallel_search(queries: list[str], api_key: str, max_results: int = 4) -> list[dict]:
    """
    Run several searches AT THE SAME TIME.

    Five sequential searches at ~2s each is ~10s. In parallel it's ~2s — the
    time of the slowest one. Searches are I/O-bound (mostly waiting on the
    network), so threads work well here.
    """
    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futures = [pool.submit(tavily_search, q, api_key, max_results) for q in queries]
        results = []
        for f in futures:
            results.extend(f.result())
    return results


# =============================================================================
#  WORDPRESS
# =============================================================================

def wp_auth(username: str, app_password: str) -> HTTPBasicAuth:
    return HTTPBasicAuth(username, app_password)


def verify_wordpress(wp_url: str, username: str, app_password: str) -> dict:
    """Check credentials before we spend time and quota. Failing fast is kinder."""
    try:
        r = requests.get(
            f"{wp_url}/wp-json/wp/v2/users/me",
            auth=wp_auth(username, app_password),
            timeout=15,
        )
    except requests.RequestException as e:
        raise AgentError(f"Couldn't reach {wp_url}. Check the URL. ({e})") from e

    if r.status_code == 200:
        return {"ok": True, "name": r.json().get("name", username)}

    if r.status_code in (401, 403):
        raise AgentError(
            "WordPress rejected those credentials. Check the username and the "
            "application password. If you run a security plugin like Wordfence, "
            "it may be blocking REST API authentication — see the guide."
        )

    raise AgentError(f"WordPress returned {r.status_code}: {r.text[:200]}")


def get_category_id(name: str, wp_url: str, auth) -> int | None:
    """Resolve a category name to its ID. Returns None if there's no match."""
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
        log.warning("category lookup failed: %s", e)
    return None


def get_internal_links(wp_url: str, auth, limit: int = 20) -> list[dict]:
    """
    The user's published posts, so the writer can link back to them.

    Internal links are one of the few SEO levers you fully control. Drafts are
    excluded — they have no public URL.
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
        log.warning("couldn't fetch internal links: %s", e)
        return []


def publish_draft(
    wp_url: str, username: str, app_password: str,
    title: str, body_markdown: str, meta_description: str, category: str,
) -> dict:
    """Push to WordPress as a DRAFT. Never publishes — a human always reviews."""
    auth = wp_auth(username, app_password)

    # markdown → HTML. This also turns [text](url) into real <a> anchors, which
    # is what makes the SEO linking work on the published page.
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
        raise AgentError(f"Couldn't reach WordPress. ({e})") from e

    if r.status_code == 201:
        post_id = r.json().get("id")
        return {
            "post_id": post_id,
            "edit_url": f"{wp_url}/wp-admin/post.php?post={post_id}&action=edit",
            "category_matched": category_id is not None,
        }

    raise AgentError(f"WordPress error {r.status_code}: {r.text[:250]}")


# =============================================================================
#  SCOUT — niche-aware
# =============================================================================

def generate_search_angles(niche: str, google_key: str, mode: str = "fast") -> list[str]:
    """
    Work out what to actually search for, given the user's niche.

    A finance blog and a cooking blog need genuinely different angles. Rather
    than substituting the niche into fixed templates, we let the model decide —
    it knows "central bank policy" matters for one and "restaurant openings"
    for the other.
    """
    prompt = f"""You are a news editor planning today's research for a blog.

THE BLOG COVERS: {niche}

Write FIVE web search queries that would surface what's genuinely newsworthy in
this space right now. Think like an editor: you want recent developments, new
launches, notable numbers, controversies, and shifts — not evergreen how-tos.

Make the queries specific and varied. They should not overlap.

Return ONLY a valid JSON array of 5 strings, no markdown fences:
["query one", "query two", "query three", "query four", "query five"]"""

    angles = parse_json_response(call_llm(prompt, google_key, mode))

    if not isinstance(angles, list) or not angles:
        raise AgentError(
            "Couldn't work out what to search for. Try describing your niche "
            "differently — be specific, like 'plant-based nutrition' rather than 'food'."
        )

    return [str(a) for a in angles[:5]]


def scout_topics(
    niche: str, tavily_key: str, google_key: str, mode: str = "fast"
) -> list[dict]:
    """Work out what to search, search it in parallel, then rank the best ideas."""
    angles = generate_search_angles(niche, google_key, mode)
    log.info("scouting angles: %s", angles)

    results = parallel_search(angles, tavily_key, max_results=4)

    if not results:
        raise AgentError(
            "Couldn't find any news on that. Try a broader description of your niche."
        )

    scan = "\n\n".join(f"- {r['title']}: {r['content'][:400]}" for r in results)

    prompt = f"""You are a ruthless editor. Below is a scan of current news. Pick the
FIVE strongest blog post ideas for a blog that covers: {niche}

Judge each on:
- FRESHNESS: genuinely new, or already everywhere?
- TRAFFIC POTENTIAL: would this blog's readers search for or click this?
- CROWDEDNESS: is every outlet already covering it? (less crowded is better)
- HOOK STRENGTH: is there a surprising number, angle, or tension to open with?
- FIT: does it genuinely belong on a blog about {niche}?

Reject generic ideas. Reject anything off-topic for this blog, however
interesting in itself. Favour specific, timely angles with a real news peg.

Return ONLY valid JSON — exactly 5 objects, no markdown fences:
[
  {{
    "topic": "the specific topic, as a clear phrase",
    "why": "one sentence on why this could pull traffic",
    "hook": "the surprising fact or tension to open with",
    "category": "a one-word category that fits this blog"
  }}
]

NEWS SCAN:
{scan[:6000]}
"""
    return parse_json_response(call_llm(prompt, google_key, mode))


# =============================================================================
#  WRITING PIPELINE
# =============================================================================

def generate_seo(topic: str, google_key: str, mode: str = "fast") -> dict:
    """Turn a topic into an SEO title, primary keyword, and meta description."""
    prompt = f"""You are an SEO strategist for a blog.

TOPIC: {topic}

Return ONLY valid JSON, no markdown fences:
{{
  "seo_title": "compelling, search-friendly title, 30-60 characters",
  "keyword": "the single primary keyword phrase to target",
  "meta_description": "120-160 character meta description containing the keyword"
}}"""
    return parse_json_response(call_llm(prompt, google_key, mode))


def research_topic(topic: str, tavily_key: str) -> dict:
    """
    Research the topic from three angles at once.

    Parallel beats sequential: we get the news, the analysis, and the numbers in
    the time of the slowest one.
    """
    queries = [topic, f"{topic} analysis", f"{topic} data statistics"]
    results = parallel_search(queries, tavily_key, max_results=3)

    if not results:
        raise AgentError("No research results found. Try a different topic.")

    # De-duplicate — the angles overlap, so the same article often appears twice.
    seen, unique = set(), []
    for r in results:
        if r.get("url") and r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)

    sources = [
        {"title": r["title"], "url": r["url"], "snippet": r["content"][:300]}
        for r in unique
    ]
    research_text = "\n\n".join(
        f"- {r['title']} (source: {r['url']}): {r['content']}" for r in unique
    )

    return {"research": research_text, "sources": sources}


def write_article(
    seo_title: str, keyword: str, research: str,
    sources: list[dict], internal_links: list[dict],
    google_key: str, mode: str = "fast",
) -> str:
    """Write the article with editorial craft and inline SEO links."""
    external = "\n".join(f"- {s['title']} → {s['url']}" for s in sources) or "(none)"
    internal = (
        "\n".join(f"- {p['title']} → {p['url']}" for p in internal_links)
        or "(none available — skip internal links)"
    )

    prompt = f"""You are a senior features writer for a publication like The Wall Street
Journal or Forbes. Your readers are intelligent and busy. They stop reading the
instant you become generic.

TITLE (first line, no markdown symbols): {seo_title}
PRIMARY KEYWORD (weave in naturally, never stuff): {keyword}

=== CRAFT RULES ===

1. THE LEDE. Open with something CONCRETE and SPECIFIC from the research — a
   startling number, a named person doing a specific thing, a sharp contrast.
   FORBIDDEN: "In today's rapidly evolving...", "In the age of...",
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
  • If none fit, use none. Forced links hurt.
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
    return call_llm(prompt, google_key, mode).strip()


def write_linkedin_post(seo_title: str, article: str, google_key: str, mode: str = "fast") -> str:
    """A LinkedIn hook post that funnels readers to the article."""
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
    return call_llm(prompt, google_key, mode).strip()


def split_title_body(draft: str) -> tuple[str, str]:
    """First line is the title; the rest is the body."""
    lines = draft.strip().split("\n", 1)
    title = lines[0].lstrip("# ").strip().strip("*").strip()
    body = lines[1].strip() if len(lines) > 1 else ""
    return title, body