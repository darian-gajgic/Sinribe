"""The shape of a generated summary, and the normalisation that makes it safe to render.

Two readers consume these definitions. The model gets them as JSON Schema, and on both paths the
decoder enforces it -- `output_config.format` on the Claude path, grammar-constrained sampling
(`format`) on the Ollama path -- so the happy path really does come back conforming. The renderer
gets the normalised dicts, and it has to survive the unhappy path as well: a 4B model held to a
schema still returns empty strings, twenty list items where four were asked for, and the occasional
null in a string field. So everything here clamps instead of validating. A thin section renders a
worse page; a KeyError renders no page at all.

Deliberately absent from the schemas: minItems/maxItems. Neither decoder is guaranteed to honour
them, and a constraint that is sometimes enforced and sometimes not is worse than one enforced in
exactly one place. Counts are asked for in the prompt and imposed here.
"""

from __future__ import annotations

import re

# Optional extra boxes a section may carry next to its summary prose, and the CSS class each maps
# to in render/webpage.py. The first version of this feature forced six forecasting lenses
# (forecast, why, now, risk, impact, question) onto every topic, copied from one hand-built page
# about one AI-forecasting interview. That reads well for exactly that kind of recording and
# badly for a tutorial or a history podcast, where "forecast" and "risk" have nothing to hold.
# These are general: each is something a learner wants pulled out of the flow of the argument.
BLOCK_KINDS = ("example", "evidence", "forecast", "caveat", "howto", "definition")
STAT_TONES = ("blue", "red", "green", "violet", "neutral")
# What the web check can conclude about one claim, worst first. "outdated" is the case this
# feature exists for: true, or at least reasonable, when it was said, and no longer true now.
# "open" is a prediction whose date has not arrived; it is reported with the latest evidence
# rather than forced into true or false. A resolved prediction is confirmed or incorrect.
VERDICT_STATUSES = ("outdated", "incorrect", "disputed", "open", "confirmed", "unverified")
# The statuses a reader must be warned about where the claim appears.
VERDICT_WARNINGS = ("outdated", "incorrect", "disputed")

# Hard ceilings. The page is meant to be read in ten to fifteen minutes; a model asked for "a
# thorough summary" will otherwise cheerfully produce thirty sections and bury the point.
MAX_SECTIONS = 16
MAX_STATS = 5
MAX_TAKEAWAYS = 6
MAX_PARAGRAPHS = 5
MAX_KEY_POINTS = 7
MAX_BLOCKS = 3
MAX_QUOTES = 3
MAX_BULLETS = 8
MAX_ROWS = 12
MAX_COLUMNS = 5
MAX_CLAIMS = 20
MAX_DEVELOPMENTS = 8
MAX_GLOSSARY = 14
MAX_QUIZ = 8
MAX_WORTH_WATCHING = 4
MAX_SOURCE_GROUPS = 6

_URL = re.compile(r"https?://[^\s<>\"'\])}]+")

# Every key name any of these schemas uses. Only needed by the leak guard below, which is
# restricted to them so it can never cut a legitimate sentence in half.
_KEYS = (
    "title", "kicker", "subtitle", "gist", "takeaways", "why_it_matters", "stats", "value",
    "label", "tone", "sections", "heading", "subheading", "focus", "starts_at", "claims", "claim",
    "time_sensitive", "section", "paragraphs", "key_points", "blocks", "kind", "bullets",
    "quotes", "text", "who", "at", "table", "caption", "columns", "rows", "recommendation",
    "overall", "checks", "status", "now", "as_of", "developments", "headline", "detail", "date",
    "glossary", "term", "meaning", "quiz", "question", "answer", "worth_watching", "why",
    "sources", "group", "links",
    "url", "footer", "topics", "numbers",
)
# A small model sometimes writes the *typographic* quote inside a JSON string. The string then
# never closes where it meant to and swallows every key that should have followed, so the answer
# parses cleanly and arrives as one field containing the rest of the object. Measured on
# gemma3:4b: a section heading came back as
#     Cochetto's Timeline", "subheading": "Understanding the Forecasted Arrival Date", "focus": ...
# Cutting at the leak recovers the field the model actually meant, and the swallowed keys fall
# back to their defaults, which is how a garbled plan still yields a usable heading.
_JSON_LEAK = re.compile(
    r'["\u201c\u201d\u2018\u2019]?\s*,\s*["\u201c\u201d\u2018\u2019]'
    r'(?:' + "|".join(_KEYS) + r')["\u201c\u201d\u2018\u2019]\s*:')


# Dashes. The prompt asks the model not to use en or em dashes, and a small model ignores the
# request about half the time -- measured on gemma3:4b, which wrote "in all domains - and the
# need" in its first section. An instruction a model can decline is not a guarantee, so the rule
# is enforced here instead: between digits a dash is a range and becomes a hyphen, spaced it is
# doing a comma's job and becomes one, and anything left becomes a hyphen. The character class
# is U+2012..U+2015 plus U+2212, the same set the check `grep -P` looks for.
_DASH = "\u2012-\u2015\u2212"
_DASH_ENTITY = re.compile(r"&[mn]dash;")
_DASH_RANGE = re.compile(rf"(?<=\d)\s*[{_DASH}]\s*(?=\d)")
_DASH_SPACED = re.compile(rf"\s*[{_DASH}]\s+")
_DASH_ANY = re.compile(rf"[{_DASH}]")


def dedash(text: str) -> str:
    """Replace en and em dashes with the punctuation that does the same job without the tell."""
    # Entities first, converted to the character rather than straight to a hyphen, so the rules
    # below get to decide what each occurrence should actually become.
    out = _DASH_ENTITY.sub("\u2014", text)
    out = _DASH_RANGE.sub("-", out)
    # A comma after a comma, or after any punctuation already doing the work, would read worse
    # than the dash did.
    out = _DASH_SPACED.sub(lambda m: " " if out[:m.start()].rstrip().endswith(
        (",", ";", ":", ".", "!", "?")) else ", ", out)
    return _DASH_ANY.sub("-", out)


def _clip(text: str, limit: int) -> str:
    """Cut `text` to `limit` without ending mid-word.

    A hard slice is how a real page came to end with "would not press a button that s". Prefer
    the last sentence boundary, which reads as if nothing was cut; fall back to the last word
    boundary with an ellipsis, which at least reads as a sentence that was cut rather than a
    typo. The 60% floor stops a field whose only full stop is near the start from losing most of
    its content to a tidy ending.
    """
    if len(text) <= limit:
        return text
    head = text[:limit]
    stop = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if stop >= limit * 0.6:
        return head[:stop + 1]
    space = head.rfind(" ")
    return (head[:space] if space > 0 else head).rstrip(" ,;:") + "\u2026"


def _s(value: object, limit: int = 4000) -> str:
    """One field as a string: never None, never ragged whitespace, never unbounded, never a
    swallowed copy of the rest of the object (see `_JSON_LEAK`)."""
    if value is None or isinstance(value, (dict, list)):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    head, sep, _rest = _JSON_LEAK.split(text, 1)[0], "", ""
    if head != text:
        # Only a string that was actually cut gets its dangling punctuation tidied. Doing this
        # unconditionally would eat the closing quote off every verbatim quotation in the page.
        head = head.rstrip().rstrip(',"\u201c\u201d\u2018\u2019').rstrip()
    return _clip(dedash(head + sep).strip(), limit)


def _lines(value: object, limit: int, each: int = 4000) -> list[str]:
    """One field as a list of non-empty strings, capped at `limit` entries."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out = [_s(v, each) for v in value]
    return [t for t in out if t][:limit]


def _one_of(value: object, allowed: tuple[str, ...], fallback: str) -> str:
    got = _s(value, 40).lower()
    return got if got in allowed else fallback


# --------------------------------------------------------------------------- schemas

def _obj(props: dict, required: list[str] | None = None) -> dict:
    """A closed object schema. `additionalProperties: false` plus a full `required` list is what
    strict structured output demands, so every field is mandatory and emptiness is expressed by an
    empty string or list rather than by omitting the key."""
    return {
        "type": "object",
        "properties": props,
        "required": required if required is not None else list(props),
        "additionalProperties": False,
    }


_STR = {"type": "string"}
_STRS = {"type": "array", "items": {"type": "string"}}
_INT = {"type": "integer"}
_LINKS = {"type": "array", "items": _obj({"label": _STR, "url": _STR})}

_PLAN_PROPS = {
    "title": _STR,
    "kicker": _STR,
    "subtitle": _STR,
    # The one-minute version: a paragraph that gives the whole idea, then the takeaways a reader
    # should leave with. Together they are the first thing on the page and the part most
    # readers will read in full.
    "gist": _STR,
    "takeaways": _STRS,
    "why_it_matters": _STR,
    "stats": {"type": "array", "items": _obj({
        "value": _STR,
        "label": _STR,
        "tone": {"type": "string", "enum": list(STAT_TONES)},
    })},
    "sections": {"type": "array", "items": _obj({
        "heading": _STR,
        "subheading": _STR,
        "focus": _STR,
        "starts_at": _STR,
    })},
}
PLAN_SCHEMA = _obj(_PLAN_PROPS)
# The same plan with the claims the web check should look at. Extracted here, in the pass that
# has just read the whole recording, rather than by the research pass, which never sees the
# transcript and so could only check what the outline happens to mention.
PLAN_WITH_CLAIMS_SCHEMA = _obj(_PLAN_PROPS | {
    "claims": {"type": "array", "items": _obj({
        "claim": _STR,
        "at": _STR,
        "section": _INT,
        "time_sensitive": {"type": "boolean"},
    })},
})

SECTION_SCHEMA = _obj({
    "heading": _STR,
    "subheading": _STR,
    "gist": _STR,
    "paragraphs": _STRS,
    "key_points": _STRS,
    "blocks": {"type": "array", "items": _obj({
        "kind": {"type": "string", "enum": list(BLOCK_KINDS)},
        "heading": _STR,
        "paragraphs": _STRS,
        "bullets": _STRS,
    })},
    "quotes": {"type": "array", "items": _obj({
        "text": _STR,
        "who": _STR,
        "at": _STR,
    })},
    "table": _obj({
        "caption": _STR,
        "columns": _STRS,
        "rows": {"type": "array", "items": _STRS},
    }),
    "recommendation": _obj({
        "paragraphs": _STRS,
        "bullets": _STRS,
    }),
})

VERDICTS_SCHEMA = _obj({
    "overall": _STR,
    "checks": {"type": "array", "items": _obj({
        "claim": _STR,
        "at": _STR,
        "section": _INT,
        "status": {"type": "string", "enum": list(VERDICT_STATUSES)},
        "now": _STR,
        "as_of": _STR,
        "sources": _LINKS,
    })},
    "developments": {"type": "array", "items": _obj({
        "headline": _STR,
        "detail": _STR,
        "date": _STR,
        "section": _INT,
        "sources": _LINKS,
    })},
})

CLOSE_SCHEMA = _obj({
    "glossary": {"type": "array", "items": _obj({"term": _STR, "meaning": _STR})},
    "quiz": {"type": "array", "items": _obj({"question": _STR, "answer": _STR})},
    "worth_watching": {"type": "array", "items": _obj({"at": _STR, "title": _STR, "why": _STR})},
    "sources": {"type": "array", "items": _obj({
        "group": _STR,
        "links": _LINKS,
    })},
    "footer": _STR,
})

NOTES_SCHEMA = _obj({
    "topics": _STRS,
    "claims": _STRS,
    "numbers": _STRS,
    "quotes": _STRS,
})


# --------------------------------------------------------------------------- normalisation

_TIMESTAMP = re.compile(r"\[?\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*\]?")


def timestamp(value: object) -> str:
    """A model-written timestamp as h:mm:ss or m:ss, or "" when it is not one."""
    match = _TIMESTAMP.search(_s(value, 40))
    return match.group(1) if match else ""


def _int(value: object, low: int, high: int) -> int:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return number if low <= number <= high else 0


def plan(raw: dict) -> dict:
    """Clamp a plan response. An empty `sections` list is the caller's cue that this failed."""
    raw = raw if isinstance(raw, dict) else {}
    stats = []
    for item in (raw.get("stats") or [])[:MAX_STATS]:
        if not isinstance(item, dict):
            continue
        value, label = _s(item.get("value"), 24), _s(item.get("label"), 220)
        if value and label:
            stats.append({"value": value, "label": label,
                          "tone": _one_of(item.get("tone"), STAT_TONES, "neutral")})

    sections = []
    for item in (raw.get("sections") or [])[:MAX_SECTIONS]:
        if not isinstance(item, dict):
            continue
        heading = _s(item.get("heading"), 140)
        if heading:
            sections.append({"heading": heading,
                             "subheading": _s(item.get("subheading"), 400),
                             "focus": _s(item.get("focus"), 700),
                             "starts_at": timestamp(item.get("starts_at"))})

    claims = []
    for item in (raw.get("claims") or [])[:MAX_CLAIMS]:
        if not isinstance(item, dict):
            continue
        claim = _s(item.get("claim"), 500)
        if claim:
            claims.append({"claim": claim, "at": timestamp(item.get("at")),
                           "section": _int(item.get("section"), 1, len(sections)),
                           "time_sensitive": bool(item.get("time_sensitive"))})

    return {
        "title": _s(raw.get("title"), 200),
        "kicker": _s(raw.get("kicker"), 160),
        "subtitle": _s(raw.get("subtitle"), 600),
        "gist": _s(raw.get("gist"), 1400),
        "takeaways": _lines(raw.get("takeaways"), MAX_TAKEAWAYS, 400),
        "why_it_matters": _s(raw.get("why_it_matters"), 1200),
        "stats": stats,
        "sections": sections,
        "claims": claims,
    }


def section(raw: dict, planned: dict, number: int) -> dict:
    """Clamp one section response, falling back to the plan's heading if the fill lost it."""
    raw = raw if isinstance(raw, dict) else {}
    blocks = []
    for item in (raw.get("blocks") or [])[:MAX_BLOCKS]:
        if not isinstance(item, dict):
            continue
        paragraphs = _lines(item.get("paragraphs"), 3, 1800)
        bullets = _lines(item.get("bullets"), MAX_BULLETS, 600)
        if not paragraphs and not bullets:
            continue
        kind = _one_of(item.get("kind"), BLOCK_KINDS, "example")
        blocks.append({
            "kind": kind,
            "heading": _s(item.get("heading"), 80) or kind.title(),
            "paragraphs": paragraphs,
            "bullets": bullets,
        })

    quotes = []
    for item in (raw.get("quotes") or [])[:MAX_QUOTES]:
        if not isinstance(item, dict):
            continue
        text = _s(item.get("text"), 600)
        if text:
            quotes.append({"text": text, "who": _s(item.get("who"), 120),
                           "at": timestamp(item.get("at")) or _s(item.get("at"), 16)})

    return {
        "id": f"s{number}",
        "number": number,
        "heading": _s(raw.get("heading"), 140) or planned.get("heading", f"Section {number}"),
        "subheading": _s(raw.get("subheading"), 400) or planned.get("subheading", ""),
        "starts_at": planned.get("starts_at", ""),
        "gist": _s(raw.get("gist"), 500),
        "paragraphs": _lines(raw.get("paragraphs"), MAX_PARAGRAPHS, 2400),
        "key_points": _lines(raw.get("key_points"), MAX_KEY_POINTS, 500),
        "blocks": blocks,
        "quotes": quotes,
        "table": _table(raw.get("table")),
        "recommendation": _recommendation(raw.get("recommendation")),
        "updates": [],
    }


def has_content(built: dict) -> bool:
    return bool(built.get("paragraphs") or built.get("key_points") or built.get("blocks"))


def _table(raw: object) -> dict | None:
    """A table only survives if it has a header row and at least one data row of matching width.

    Ragged tables are the single most common structural failure here: the model settles on four
    columns, then emits a three-cell row. Rather than render a broken grid, short rows are padded
    and long ones truncated to the header's width.
    """
    if not isinstance(raw, dict):
        return None
    columns = _lines(raw.get("columns"), MAX_COLUMNS, 80)
    if len(columns) < 2:
        return None
    rows = []
    for row in (raw.get("rows") or [])[:MAX_ROWS]:
        cells = _lines(row, MAX_COLUMNS, 400) if isinstance(row, list) else []
        if not any(cells):
            continue
        cells = (cells + [""] * len(columns))[:len(columns)]
        rows.append(cells)
    if not rows:
        return None
    return {"caption": _s(raw.get("caption"), 220), "columns": columns, "rows": rows}


def _recommendation(raw: object) -> dict | None:
    if not isinstance(raw, dict):
        return None
    paragraphs = _lines(raw.get("paragraphs"), 3, 2200)
    bullets = _lines(raw.get("bullets"), MAX_BULLETS, 700)
    if not paragraphs and not bullets:
        return None
    return {"paragraphs": paragraphs, "bullets": bullets}


def _links(raw: object, allowed: set[str], limit: int = 24) -> list[dict]:
    """Links that survive the allow-list, deduplicated, labelled."""
    out = []
    seen: set[str] = set()
    for link in (raw if isinstance(raw, list) else [])[:limit]:
        if not isinstance(link, dict):
            continue
        url = _s(link.get("url"), 500)
        key = _url_key(url)
        if not url or key in seen or not _url_allowed(url, allowed):
            continue
        seen.add(key)
        out.append({"label": _s(link.get("label"), 140) or _host(url), "url": url})
    return out


def verdicts(raw: dict, allowed_urls: set[str], sections: int) -> dict:
    """Clamp the web check, and refuse any verdict the search results do not back.

    A verdict is only as good as its source. "Outdated" with no link is the model's opinion
    presented as a finding, which on a page whose job is telling the reader what changed is worse
    than saying nothing. So every non-"unverified" verdict needs at least one link that the
    search tools actually returned; one without is downgraded to "unverified" and says why. A
    development with no surviving source is dropped outright: it is not tied to anything the
    reader said or heard, so there is nothing left to show.
    """
    raw = raw if isinstance(raw, dict) else {}
    checks = []
    for item in (raw.get("checks") or [])[:MAX_CLAIMS]:
        if not isinstance(item, dict):
            continue
        claim = _s(item.get("claim"), 500)
        if not claim:
            continue
        status = _one_of(item.get("status"), VERDICT_STATUSES, "unverified")
        links = _links(item.get("sources"), allowed_urls, 6)
        now = _s(item.get("now"), 900)
        if status != "unverified" and not links:
            status = "unverified"
            now = ("No source the search returned backs a verdict on this."
                   + (f" The check's note: {now}" if now else ""))
        checks.append({"claim": claim, "at": timestamp(item.get("at")),
                       "section": _int(item.get("section"), 1, sections),
                       "status": status, "now": now, "as_of": _date(item.get("as_of")),
                       "sources": links})
    # Worst news first: that is what a reader opened this section for.
    checks.sort(key=lambda c: VERDICT_STATUSES.index(c["status"]))

    developments = []
    for item in (raw.get("developments") or [])[:MAX_DEVELOPMENTS]:
        if not isinstance(item, dict):
            continue
        headline = _s(item.get("headline"), 200)
        links = _links(item.get("sources"), allowed_urls, 4)
        if headline and links:
            developments.append({"headline": headline, "detail": _s(item.get("detail"), 900),
                                 "date": _date(item.get("date")),
                                 "section": _int(item.get("section"), 1, sections),
                                 "sources": links})
    return {"overall": _s(raw.get("overall"), 900), "checks": checks,
            "developments": developments}


def _date(value: object) -> str:
    """A model-written date, or "" when it is not one. Measured: a development came back with
    "Various; version numbers from docs and..." in its date field, which the page then printed
    in the date column."""
    text = _s(value, 40)
    return text if re.search(r"\b(19|20)\d{2}\b", text) and len(text) <= 32 else ""


def close(raw: dict, allowed_urls: set[str]) -> dict:
    """Clamp the closing pass. Every link is checked against `allowed_urls`.

    This is the one place where a wrong answer is actively harmful rather than merely thin: an
    invented citation looks exactly like a real one and quietly turns the page from a summary into
    a source of false provenance. So links are not trusted, they are matched -- a URL that did not
    come back from a web search or appear in the recording's own material is dropped, and a group
    left with no links is dropped with it.
    """
    raw = raw if isinstance(raw, dict) else {}
    glossary = []
    for item in (raw.get("glossary") or [])[:MAX_GLOSSARY]:
        if isinstance(item, dict):
            term, meaning = _s(item.get("term"), 80), _s(item.get("meaning"), 600)
            if term and meaning:
                glossary.append({"term": term, "meaning": meaning})
    quiz = []
    for item in (raw.get("quiz") or [])[:MAX_QUIZ]:
        if isinstance(item, dict):
            question, answer = _s(item.get("question"), 300), _s(item.get("answer"), 700)
            if question and answer:
                quiz.append({"question": question, "answer": answer})

    watch = []
    for item in (raw.get("worth_watching") or [])[:MAX_WORTH_WATCHING]:
        if isinstance(item, dict):
            at, title = timestamp(item.get("at")), _s(item.get("title"), 160)
            if at and title:
                watch.append({"at": at, "title": title, "why": _s(item.get("why"), 500)})

    groups = []
    for item in (raw.get("sources") or [])[:MAX_SOURCE_GROUPS]:
        if not isinstance(item, dict):
            continue
        links = _links(item.get("links"), allowed_urls)
        if links:
            groups.append({"group": _s(item.get("group"), 80) or "Sources", "links": links})

    return {"glossary": glossary, "quiz": quiz, "worth_watching": watch, "sources": groups,
            "footer": _s(raw.get("footer"), 900)}


_MD_LINK = re.compile(r"\[([^\]\[]{1,160})\]\((https?://[^\s)]{1,500})\)")


def restrict_links(value: object, allowed_urls: set[str]) -> object:
    """Unlink every [label](url) in generated text whose URL is not on the allow-list.

    The Sources list was always checked; links written inside the prose were not, so a section
    could cite a URL nothing ever returned and the page would render it as a live link. The label
    is kept, so the sentence still reads; only the false provenance goes. Applied to the whole
    summary, except the `url` fields that the source and verdict checks above already filtered.
    """
    if isinstance(value, str):
        return _MD_LINK.sub(lambda m: m.group(0) if _url_allowed(m.group(2), allowed_urls)
                            else m.group(1), value)
    if isinstance(value, list):
        return [restrict_links(v, allowed_urls) for v in value]
    if isinstance(value, dict):
        return {k: (v if k == "url" else restrict_links(v, allowed_urls))
                for k, v in value.items()}
    return value


def _url_key(url: str) -> str:
    """A URL reduced to what identifies the page: no scheme, no www., no fragment, no trailing
    slash, lower case. The model often cites a search hit with one of those changed."""
    key = url.split("://", 1)[-1].split("#", 1)[0].rstrip("/").lower()
    return key[4:] if key.startswith("www.") else key


def _url_allowed(url: str, allowed: set[str]) -> bool:
    """Is this link one we actually saw?"""
    return _url_key(url) in {_url_key(a) for a in allowed}


def _host(url: str) -> str:
    return url.split("://", 1)[-1].split("/", 1)[0] or url


def notes(raw: dict) -> dict:
    """Clamp one chunk's notes. Used only on the small-context path."""
    raw = raw if isinstance(raw, dict) else {}
    return {
        "topics": _lines(raw.get("topics"), 6, 200),
        "claims": _lines(raw.get("claims"), 8, 500),
        "numbers": _lines(raw.get("numbers"), 6, 200),
        "quotes": _lines(raw.get("quotes"), 4, 400),
    }


def find_urls(text: str) -> set[str]:
    """Every URL in a blob of text, with trailing sentence punctuation stripped.

    Feeds the citation guard in `close`: the allowed set is built from the research transcript and
    the media source, so it has to see links exactly as the model will later cite them.
    """
    found = set()
    for raw in _URL.findall(text or ""):
        found.add(raw.rstrip(".,;:!?'\""))
    return found
