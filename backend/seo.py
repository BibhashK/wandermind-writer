"""
On-page SEO scorer.

HONESTY NOTE: This is an ON-PAGE score, not a ranking prediction. A true SEO
score would need keyword search-volume and competition data (Ahrefs/Semrush —
paid APIs). What we measure here is what we can actually compute from the text
itself: the on-page fundamentals. These are real, well-established factors, and
every check below is transparent so the user can see exactly why they scored
what they scored.
"""

import re
from typing import TypedDict


class SEOCheck(TypedDict):
    label: str      # human-readable name of the check
    passed: bool    # did it pass?
    detail: str     # what we found
    weight: int     # how many points this check is worth


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _strip_markdown(text: str) -> str:
    """Remove markdown syntax so we count real prose, not symbols."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)   # links → anchor text
    text = re.sub(r"[#*_`>]", "", text)                     # heading/bold marks
    return text


def score_article(
    title: str,
    body_markdown: str,
    keyword: str,
    meta_description: str,
) -> dict:
    """
    Score an article on on-page SEO fundamentals.

    Returns a dict with the total score (0-100), a letter grade, and the full
    breakdown so the user can see exactly what passed and what didn't.
    """
    checks: list[SEOCheck] = []

    kw = keyword.lower().strip()
    title_l = title.lower()
    body_l = body_markdown.lower()
    prose = _strip_markdown(body_markdown)
    word_count = _count_words(prose)

    # --- 1. Keyword in the title (high value) ---------------------------------
    kw_in_title = kw in title_l
    checks.append({
        "label": "Keyword in title",
        "passed": kw_in_title,
        "detail": f"'{keyword}' {'found' if kw_in_title else 'missing'} in title",
        "weight": 15,
    })

    # --- 2. Title length ------------------------------------------------------
    # Google truncates titles around 60 characters.
    title_len = len(title)
    title_ok = 30 <= title_len <= 60
    checks.append({
        "label": "Title length",
        "passed": title_ok,
        "detail": f"{title_len} chars (ideal: 30-60)",
        "weight": 10,
    })

    # --- 3. Keyword in the first paragraph ------------------------------------
    first_chunk = " ".join(prose.split()[:100]).lower()
    kw_early = kw in first_chunk
    checks.append({
        "label": "Keyword appears early",
        "passed": kw_early,
        "detail": "in first 100 words" if kw_early else "not in first 100 words",
        "weight": 10,
    })

    # --- 4. Meta description length -------------------------------------------
    meta_len = len(meta_description)
    meta_ok = 120 <= meta_len <= 160
    checks.append({
        "label": "Meta description length",
        "passed": meta_ok,
        "detail": f"{meta_len} chars (ideal: 120-160)",
        "weight": 10,
    })

    # --- 5. Keyword in meta description ---------------------------------------
    kw_in_meta = kw in meta_description.lower()
    checks.append({
        "label": "Keyword in meta description",
        "passed": kw_in_meta,
        "detail": "found" if kw_in_meta else "missing",
        "weight": 5,
    })

    # --- 6. Word count --------------------------------------------------------
    # Long enough to be substantive; not so long it's padded.
    wc_ok = 600 <= word_count <= 2000
    checks.append({
        "label": "Article length",
        "passed": wc_ok,
        "detail": f"{word_count} words (ideal: 600-2000)",
        "weight": 10,
    })

    # --- 7. Subheadings -------------------------------------------------------
    headings = re.findall(r"^#{2,3}\s+.+$", body_markdown, flags=re.MULTILINE)
    headings_ok = len(headings) >= 3
    checks.append({
        "label": "Subheadings (H2/H3)",
        "passed": headings_ok,
        "detail": f"{len(headings)} found (want 3+)",
        "weight": 10,
    })

    # --- 8. Keyword density ---------------------------------------------------
    # Present, but not stuffed. Roughly 0.5%-2.5% is the healthy band.
    kw_count = body_l.count(kw) if kw else 0
    density = (kw_count / word_count * 100) if word_count else 0
    density_ok = 0.5 <= density <= 2.5
    checks.append({
        "label": "Keyword density",
        "passed": density_ok,
        "detail": f"{density:.1f}% ({kw_count}x) — ideal 0.5-2.5%",
        "weight": 10,
    })

    # --- 9. External links (credibility signal) -------------------------------
    all_links = re.findall(r"\[([^\]]+)\]\((https?://[^)]+)\)", body_markdown)
    ext_links = len(all_links)
    ext_ok = ext_links >= 2
    checks.append({
        "label": "Outbound source links",
        "passed": ext_ok,
        "detail": f"{ext_links} found (want 2+)",
        "weight": 10,
    })

    # --- 10. Descriptive anchor text ------------------------------------------
    bad_anchors = [a for a, _ in all_links
                   if a.lower().strip() in {"here", "click here", "link", "this"}]
    anchors_ok = len(bad_anchors) == 0 and ext_links > 0
    checks.append({
        "label": "Descriptive link anchors",
        "passed": anchors_ok,
        "detail": (
            f"{len(bad_anchors)} vague anchor(s)" if bad_anchors
            else "all descriptive" if ext_links else "no links to check"
        ),
        "weight": 10,
    })

    # --- Total ----------------------------------------------------------------
    earned = sum(c["weight"] for c in checks if c["passed"])
    possible = sum(c["weight"] for c in checks)
    score = round(earned / possible * 100) if possible else 0

    if score >= 90:
        grade, verdict = "A", "Excellent — publish it."
    elif score >= 75:
        grade, verdict = "B", "Strong. Fix the flagged items to push higher."
    elif score >= 60:
        grade, verdict = "C", "Decent, but leaving traffic on the table."
    elif score >= 40:
        grade, verdict = "D", "Weak on-page SEO. Worth a revision pass."
    else:
        grade, verdict = "F", "Needs significant work before publishing."

    return {
        "score": score,
        "grade": grade,
        "verdict": verdict,
        "word_count": word_count,
        "checks": checks,
        "passed_count": sum(1 for c in checks if c["passed"]),
        "total_count": len(checks),
    }