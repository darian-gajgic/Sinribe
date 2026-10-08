"""Turning a transcript into a brief you can learn from, one bounded question at a time.

The reader this is written for pasted a link because the recording looked worth knowing and two
hours of listening did not. So the page is built in the order that reader needs it:

    1. the one-minute version   the whole idea in a paragraph, then the takeaways
    2. what has changed         when the web check ran: which claims no longer hold, and why
    3. the full summary         one section per topic, prose first, the facts worth keeping after
    4. key terms and a quiz     what a newcomer needs to follow it, and a way to check it stuck

Getting there takes several passes rather than one big ask. A single call that has to invent the
outline AND fill it writes a strong first section and a tired last one, because it is budgeting an
unknown remaining length. And the outline is worth seeing before it is filled: once the plan
exists, each section can be written against the full transcript with the rest of the plan in
view, so sections stay distinct instead of restating each other.

    notes     only on a small-context provider: map the transcript down to what fits
    plan      title, the one-minute version, figures, the outline, and the claims to check
    research  optional, Claude only: search the web for what is true NOW about those claims
    verify    turn the research report into one verdict per claim, sources matched to the search
    sections  one call per section, the transcript cached in the prefix
    close     key terms, quiz, the grouped sources, the footer

On the Claude path the transcript sits behind a cache breakpoint (API) or in one resumed session
(Claude Code), so the dozen section calls read it from cache rather than paying for it each time.
"""

from __future__ import annotations

import datetime as _dt
from typing import Callable

from ..config import LANGUAGES
from ..merge import Turn
from ..textfmt import hhmmss, human_duration
from . import schema
from .providers import Provider, SummaryError, Unavailable

# Relative cost of each pass, for the progress bar. Sections dominate: a dozen calls against a
# cached prefix, each asking for real prose. Research waits on searches and page reads.
PASS_WEIGHTS = {"notes": 0.18, "plan": 0.12, "research": 0.18, "verify": 0.04,
                "sections": 0.42, "close": 0.06}

# Output budgets, which on the API include thinking. Every Claude call streams, so a generous
# ceiling costs nothing unless it is used; the local provider caps them at its own window.
MAX_TOKENS = {"notes": 2_000, "plan": 16_000, "research": 32_000, "verify": 16_000,
              "sections": 32_000, "close": 16_000}

# How much of the notes stage is reading the transcript, with the rest left for folding the
# notes down to size. Reading is the bigger half and the only part a short recording performs.
NOTES_READ_SHARE = 0.75

DEFAULT_SECTIONS = 12
# A 4B model writing its twelfth section has long since run out of anything to say. Fewer, fuller
# sections read better than many empty ones.
SMALL_CONTEXT_SECTIONS = 6
# Roughly how much recording one section is worth. The configured maximum is a ceiling for a
# feature-length interview, not a quota: asked for twelve sections from a two-minute clip, the
# local model returned a plan with no sections at all and the summary failed outright. Ten
# minutes a section puts a two-hour podcast at the ceiling and a short clip at the floor.
SECONDS_PER_SECTION = 600
MIN_SECTIONS = 3
# How many claims the web check looks at. Each costs roughly a search and sometimes a page read;
# sixteen covers the checkable core of a two-hour interview without turning the check into the
# most expensive part of the job.
DEFAULT_CLAIMS = 16
# How much of the uploader's description goes into the plan and research prompts.
DESCRIPTION_CHARS = 2_500
# The full summary's length, as a share of the words actually spoken. The point of the brief is
# to be faster than listening: measured on a 10m45s video, sections written without a budget came
# to about 2,900 words, longer than the 2,400-word transcript, and the page said "13 min full
# summary" next to "10m 45s listening". Fifteen percent puts a two-hour interview near the
# 3,000-word ceiling (about a quarter of an hour of reading) and a ten-minute video at a page.
SUMMARY_SHARE = 0.15
SUMMARY_MIN_WORDS = 400
SUMMARY_MAX_WORDS = 3_000
SECTION_MIN_WORDS = 150
SECTION_MAX_WORDS = 350


def _language_rule(code: str | None) -> str:
    """The 'write in X' instruction, named explicitly rather than left to inference.

    Same lesson as enrich.py: "the same language as the input" is read as a style note and
    answered in English anyway. A German interview came back with English section titles.
    """
    name = dict(LANGUAGES).get((code or "").lower())
    if not name or name == "Auto-detect":
        return "Write it in the same language the transcript is in."
    return f"Write it in {name}, the language of the transcript."


def _system(language: str | None) -> str:
    return (
        "You turn the transcript of a recorded conversation or talk into a brief for a reader "
        "who wants to learn what it teaches without watching it. After ten minutes with your "
        "brief they should understand the recording as well as someone who sat through it. "
        "You are precise, concrete and clear.\n\n"
        "Rules:\n"
        "- Ground everything in the transcript. If it does not support a claim, leave the claim "
        "out. Do not invent numbers, names, dates or sources.\n"
        "- Prefer the specific over the general: a figure the speaker gave, a date, a named "
        "company, the actual mechanism. Never write a sentence that would be true of any "
        "recording.\n"
        "- Say who holds a view. Attribute claims and opinions to the speaker by name.\n"
        "- Explain any term a newcomer would not know the first time it appears.\n"
        "- Practical guidance you derived rather than heard must start with \"(derived)\".\n"
        "- Quotes must be word for word from the transcript, and short.\n"
        f"- {_language_rule(language)}\n"
        "- Do not use em dashes or en dashes. Use a comma, a colon, or a new sentence. Vary "
        "sentence length; a page of uniform 20-word sentences reads as machine-written.\n"
        "- You may use **bold**, *italics*, `code` and [label](url) inside any text field. No "
        "other markup, and no headings.\n"
        "- Reply with JSON conforming to the schema and nothing else."
    )


def transcript_text(turns: list[Turn], limit: int = 0) -> str:
    """The transcript as timestamped speaker lines.

    Timestamps are in the text on purpose: they let the model attribute a quote to a point in the
    recording, which is what makes a quote in the summary checkable against the audio.
    """
    lines = [f"[{hhmmss(t.start)}] {t.speaker}: {t.text.replace(chr(10), ' ')}" for t in turns]
    text = "\n".join(lines)
    return text[:limit] if limit and len(text) > limit else text


def _chunks(text: str, size: int) -> list[str]:
    """Split on line boundaries so no speaker turn is cut in half."""
    out: list[str] = []
    buf: list[str] = []
    used = 0
    for line in text.splitlines():
        if buf and used + len(line) > size:
            out.append("\n".join(buf))
            buf, used = [], 0
        buf.append(line)
        used += len(line) + 1
    if buf:
        out.append("\n".join(buf))
    return out


def age_days(published: str, today: _dt.date | None = None) -> int | None:
    """Days between publication and today, or None when the date is missing or malformed."""
    try:
        when = _dt.date.fromisoformat(str(published or "")[:10])
    except ValueError:
        return None
    return max(0, ((today or _dt.date.today()) - when).days)


def _source_block(result: dict, stats: dict) -> str:
    """What we know about the recording, independent of its contents."""
    bits = [f"Title: {result.get('title') or 'unknown'}"]
    if result.get("uploader"):
        bits.append(f"Published by: {result['uploader']}")
    age = age_days(result.get("published", ""))
    if age is not None:
        bits.append(f"Published on: {result['published']} ({age} days before today)")
    else:
        bits.append("Published on: unknown")
    if result.get("source"):
        bits.append(f"Source: {result['source']}")
    if result.get("duration"):
        bits.append(f"Length: {human_duration(float(result['duration']))}")
    if stats:
        bits.append("Speakers: " + ", ".join(sorted(stats)))
    if result.get("language"):
        bits.append(f"Language: {result['language']}")
    bits.append(f"Today's date: {_dt.date.today().isoformat()}")
    return "\n".join(bits)


def _uploader_block(result: dict) -> str:
    """The uploader's chapters and description, for the passes that plan and look things up."""
    parts = []
    chapters = [c for c in result.get("chapters") or [] if isinstance(c, dict)]
    if chapters:
        parts.append("THE UPLOADER'S CHAPTER MARKERS:\n" + "\n".join(
            f"[{hhmmss(float(c.get('start') or 0))}] {c.get('title', '')}" for c in chapters))
    description = str(result.get("description") or "").strip()
    if description:
        parts.append("THE UPLOADER'S DESCRIPTION (it often lists what the episode references):\n"
                     + description[:DESCRIPTION_CHARS])
    return ("\n\n" + "\n\n".join(parts)) if parts else ""


class _Progress:
    """Spreads 0..1 across whichever passes this run actually performs."""

    def __init__(self, passes: list[str], report: Callable[[float], None] | None):
        total = sum(PASS_WEIGHTS[p] for p in passes) or 1.0
        self._spans: dict[str, tuple[float, float]] = {}
        acc = 0.0
        for name in passes:
            width = PASS_WEIGHTS[name] / total
            self._spans[name] = (acc, width)
            acc += width
        self._report = report or (lambda _: None)

    def at(self, name: str, fraction: float) -> None:
        base, width = self._spans.get(name, (0.0, 0.0))
        self._report(min(1.0, base + width * max(0.0, min(1.0, fraction))))


def _dedupe_sources(sources: list[dict]) -> list[dict]:
    """One entry per page, keeping the first title seen for it."""
    out: list[dict] = []
    seen: set[str] = set()
    for item in sources:
        url = str(item.get("url") or "")
        key = schema._url_key(url)
        if url.startswith("http") and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def summarize(
    turns: list[Turn],
    stats: dict,
    result: dict,
    provider: Provider,
    *,
    research: bool = False,
    max_sections: int = DEFAULT_SECTIONS,
    max_claims: int = DEFAULT_CLAIMS,
    on_progress: Callable[[float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """Build the structured summary. Raises SummaryError if the result would be unusable."""
    log = on_log or (lambda _: None)
    cancelled = should_cancel or (lambda: False)
    # So a provider can report a recovered or degraded answer instead of swallowing it.
    provider.on_log = log

    def check() -> None:
        if cancelled():
            raise _Cancelled()

    language = result.get("language")
    system = _system(language)
    full = transcript_text(turns)
    if not full.strip():
        raise SummaryError("the transcript is empty, so there is nothing to summarise")

    small = len(full) > provider.context_chars
    research_error = ""
    if research and not provider.supports_research:
        research_error = f"{provider.label} cannot search the web"
        log(f"WARNING: web research needs a Claude provider and {provider.label} cannot search. "
            f"The brief is written from the transcript alone.")
        research = False
    if small:
        max_sections = min(max_sections, SMALL_CONTEXT_SECTIONS)
    max_sections = _section_count(max_sections, float(result.get("duration") or 0.0))
    max_claims = max(4, min(int(max_claims or DEFAULT_CLAIMS), schema.MAX_CLAIMS))

    passes = (["notes"] if small else []) + ["plan"] \
        + (["research", "verify"] if research else []) + ["sections", "close"]
    progress = _Progress(passes, on_progress)
    source = _source_block(result, stats)
    uploader = _uploader_block(result)
    media_url = str(result.get("source") or "")
    # The links the page may print: what the recording and its uploader mention, the recording
    # itself, and (added after research) what the search tools actually returned. Nothing the
    # model merely wrote down.
    recording_urls = schema.find_urls(full) | schema.find_urls(str(result.get("description") or ""))
    allowed_urls = set(recording_urls) | ({media_url} if media_url.startswith("http") else set())

    # -- notes ---------------------------------------------------------------------
    context_label, context = "TRANSCRIPT", full
    if small:
        pieces = _chunks(full, provider.context_chars)
        log(f"transcript is {len(full) / 1000:.0f} kB and {provider.label} holds "
            f"{provider.context_chars / 1000:.0f} kB, so summarising in {len(pieces)} passes")
        notes: list[dict] = []
        for i, piece in enumerate(pieces):
            check()
            notes.append(schema.notes(provider.ask_json(
                system,
                "Take notes on this excerpt of the transcript for a later summary. List the "
                "topics discussed, the substantive claims made, every concrete number or date, "
                "and up to four short verbatim quotes worth keeping.\n\n"
                f"EXCERPT ({i + 1} of {len(pieces)}):\n{piece}",
                schema.NOTES_SCHEMA, max_tokens=MAX_TOKENS["notes"])))
            # Only three quarters of this stage's span: folding the notes below is more model
            # calls, and a bar that stops moving for several minutes reads as a hung job.
            progress.at("notes", NOTES_READ_SHARE * (i + 1) / len(pieces))
        context_label, context = "TRANSCRIPT NOTES", _digest(
            notes, provider, system, _digest_budget(provider), check, log,
            lambda f: progress.at("notes", NOTES_READ_SHARE + (1 - NOTES_READ_SHARE) * f))

    prefix = f"{context_label}:\n{context}"

    # -- plan ----------------------------------------------------------------------
    check()
    log(f"planning the summary with {provider.label}")
    plan = _plan(provider, system, source + uploader, prefix, max_sections,
                 max_claims if research else 0, check, log)
    if not plan["sections"]:
        raise SummaryError(f"{provider.label} returned no section plan for this transcript")
    progress.at("plan", 1.0)
    log(f"planned {len(plan['sections'])} sections: "
        + "; ".join(s["heading"] for s in plan["sections"][:4])
        + ("; …" if len(plan["sections"]) > 4 else ""))

    # -- research + verify -----------------------------------------------------------
    web_check: dict | None = None
    research_sources: list[dict] = []
    if research:
        check()
        try:
            web_check, research_sources = _web_check(
                provider, system, source, uploader, plan, prefix, allowed_urls, log, check,
                progress)
        except _Cancelled:
            raise
        except (SummaryError, Unavailable) as e:
            # The web check is the part of this job most exposed to the outside world: searches
            # time out, pages refuse, the CLI can die mid-run. None of that is a reason to lose
            # the brief, so the page is written from the transcript and says plainly that the
            # check did not happen.
            research_error = str(e)
            log(f"WARNING: the web check failed and the brief is written from the transcript "
                f"alone: {e}")
            web_check = None

    # -- sections ------------------------------------------------------------------
    outline = "\n".join(f"{i + 1}. {s['heading']}" for i, s in enumerate(plan["sections"]))
    # A small model asked for six blocks writes six thin ones and then overruns its output
    # window part way through the JSON. Asking it for less is what keeps the answer complete.
    budget = section_budget(len(full.split()), len(plan["sections"]))
    tight = small or budget < 220
    para_rule = ("one or two short paragraphs" if tight else "two to four paragraphs")
    boxes = "zero or one optional box" if tight else "zero to three optional boxes"
    block_rule = (
        "leave empty.\n" if small else
        f"{boxes} for material worth pulling out of the prose, each a "
        "different kind: 'example' for a concrete story or case from the recording, 'evidence' "
        "for data, studies or sources cited, 'forecast' for a prediction with its date, "
        "'caveat' for counterpoints, limits or what went unaddressed, 'howto' for practical "
        "steps or advice given, 'definition' for a concept explained at length. Give each a two "
        "to four word heading. Only use one when the recording has real material for it.\n")
    table_rule = ("- table: leave columns and rows empty.\n" if small else
                  "- table: only if this topic genuinely is a comparison (options, scenarios, a "
                  "timeline). Otherwise leave columns and rows empty.\n")
    sections = []
    planned = plan["sections"]
    for i, want in enumerate(planned):
        check()
        log(f"writing section {i + 1}/{len(planned)}: {want['heading']}")
        starts = f" It starts around [{want['starts_at']}]." if want.get("starts_at") else ""
        check_block, check_rule = _section_check(web_check, i + 1)
        raw = _try_section(
            provider, system,
            f"RECORDING:\n{source}\n\nTHE FULL OUTLINE (other sections cover these, do not "
            f"duplicate them):\n{outline}{check_block}\n\n"
            f"Write section {i + 1}, \"{want['heading']}\".{starts}\n"
            f"Its job: {want['focus'] or want['subheading']}\n\n"
            f"- gist: the point of this part in one sentence.\n"
            f"- paragraphs: {para_rule} that explain this part of the recording to a reader "
            f"who did not watch it, so they come away knowing what was said: the claims, the "
            f"reasoning behind them, the examples and the numbers. This is the main text of the "
            f"section; make it complete enough to learn from.\n"
            f"- key_points: three to six bullets with the facts, names, numbers and ideas worth "
            f"remembering from this part.\n"
            f"- blocks: {block_rule}"
            f"- quotes: up to three short verbatim quotes from this part of the transcript, with "
            f"who said it and the [h:mm:ss] timestamp in 'at'.\n"
            f"{table_rule}"
            f"- recommendation: what the reader could do with this, if the recording gives "
            f"practical advice or it follows clearly from it. Leave both lists empty when there "
            f"is nothing actionable.\n\n"
            f"Keep the whole section to about {budget} words, key points and boxes included. "
            f"The reader chose this brief to save time; cut repetition before you cut "
            f"substance.{check_rule}",
            cache_prefix=prefix, log=log, number=i + 1)
        built = schema.section(raw, want, i + 1)
        if schema.has_content(built):
            sections.append(built)
        else:
            log(f"section {i + 1} came back empty and was dropped")
        progress.at("sections", (i + 1) / len(planned))

    if not sections:
        raise SummaryError(f"{provider.label} produced no usable sections")
    if web_check:
        _attach_updates(sections, web_check)

    # -- close ---------------------------------------------------------------------
    check()
    log("writing the key terms, quiz, moments worth watching and sources")
    written = "\n".join(f"{s['number']}. {s['heading']}" for s in sections)
    link_lists = _link_lists(research_sources, recording_urls)
    source_rule = ("Group the most useful links from the lists above by theme, for a reader who "
                   "wants to go deeper. Use only URLs from those lists."
                   if research_sources else
                   "Group the links from the list above that a reader would want by theme. Use "
                   "only URLs from that list, and return an empty list if there are none.")
    close = schema.close(provider.ask_json(
        system,
        f"RECORDING:\n{source}\n\nSECTIONS WRITTEN:\n{written}{link_lists}\n\n"
        f"Close the brief.\n"
        f"- glossary: four to twelve terms, names or concepts from the recording that a "
        f"newcomer needs, each with a plain one or two sentence meaning as used in this "
        f"recording.\n"
        f"- quiz: five to eight questions that test whether the reader understood the main "
        f"ideas, each with a short answer. Ask about ideas and reasons, not trivia.\n"
        f"- worth_watching: two to four moments worth hearing in the original, because the "
        f"delivery, a demonstration or the exact wording matters more than a summary can "
        f"carry. Each with its [h:mm:ss] timestamp in 'at', a short title, and one sentence on "
        f"why.\n"
        f"- sources: {source_rule}\n"
        f"- footer: one or two sentences on what this page is and what it was made from, and "
        f"that text marked (derived) is inference rather than quotation.",
        schema.CLOSE_SCHEMA, max_tokens=MAX_TOKENS["close"], cache_prefix=prefix), allowed_urls)
    progress.at("close", 1.0)
    spent = float(getattr(provider, "cost_usd", 0.0) or 0.0)
    if spent:
        log(f"{provider.label} reported ${spent:.2f} of usage for this summary")

    if media_url.startswith("http") and not any(
            schema._url_key(link["url"]) == schema._url_key(media_url)
            for g in close["sources"] for link in g["links"]):
        # The page must always link back to what it summarises, whatever the model decided.
        close["sources"].insert(0, {"group": "Primary", "links": [
            {"label": result.get("title") or media_url, "url": media_url}]})

    brief = {
        "title": plan["title"] or str(result.get("title") or "Summary"),
        "kicker": plan["kicker"],
        "subtitle": plan["subtitle"],
        "gist": plan["gist"],
        "takeaways": plan["takeaways"],
        "why_it_matters": plan["why_it_matters"],
        "stats": plan["stats"],
        "check": web_check,
        "sections": sections,
        "glossary": close["glossary"],
        "quiz": close["quiz"],
        "worth_watching": close["worth_watching"],
        "sources": close["sources"],
        "footer": close["footer"],
    }
    # One pass over everything the model wrote: a link survives only if it is on the list.
    brief = schema.restrict_links(brief, allowed_urls)
    published = str(result.get("published") or "")
    brief["meta"] = {
        "provider": provider.name,
        # What the provider says this cost. On the Claude Code path that is subscription usage
        # rather than a bill, but a number the user can see beats a number they cannot.
        "cost_usd": round(float(getattr(provider, "cost_usd", 0.0) or 0.0), 4),
        "model": getattr(provider, "model", provider.name),
        "label": provider.label,
        "skipped": list(getattr(provider, "skipped", []) or []),
        "research": web_check is not None,
        "research_error": research_error,
        "checked_on": _dt.date.today().isoformat() if web_check is not None else "",
        "language": language,
        "source": media_url,
        "media_title": result.get("title") or "",
        "uploader": result.get("uploader") or "",
        "published": published,
        "age_days": age_days(published),
        "duration": float(result.get("duration") or 0.0),
        "speakers": sorted(stats) if stats else [],
        "generated_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    return brief


def _web_check(provider: Provider, system: str, source: str, uploader: str, plan: dict,
               prefix: str, allowed_urls: set[str], log: Callable[[str], None],
               check: Callable[[], None], progress: _Progress) -> tuple[dict, list[dict]]:
    """Search the web for what is true now, then pin one verdict to each claim.

    Two calls on purpose. The research call has the web tools and writes a report; the verify
    call has no tools and turns that report into the verdict schema. Asking one call to search
    AND hold a strict output shape gets a decoder fighting a tool loop. And the sources are taken
    from what the tools returned, never from the report's prose, so `allowed_urls` grows only by
    pages a search really produced or a fetch really opened.
    """
    log("checking the recording's claims against the web")
    claims = plan.get("claims") or []
    if not claims:
        # A plan without claims still deserves a check: its takeaways are the brief's claims.
        claims = [{"claim": t, "at": "", "section": 0, "time_sensitive": True}
                  for t in plan.get("takeaways") or []]
    listed = "\n".join(
        f"{i + 1}. {c['claim']}" + (f" [{c['at']}]" if c.get("at") else "")
        + (f" (section {c['section']})" if c.get("section") else "")
        + (" (time-sensitive)" if c.get("time_sensitive") else "")
        for i, c in enumerate(claims))
    outline = "\n".join(f"{i + 1}. {s['heading']}" for i, s in enumerate(plan["sections"]))
    report, sources = provider.research(
        "You are a careful fact-checker for a reader who learns from recorded talks and "
        "podcasts, often about fast-moving fields like AI, where a recording a few months old "
        "can already be out of date. Your job is to find out what is true TODAY. You report "
        "what sources say and name them. You never present a guess as a finding: when you "
        "cannot find good evidence, you say so. Do not use em dashes or en dashes.",
        f"RECORDING:\n{source}{uploader}\n\nWHAT IT IS ABOUT:\n{plan.get('gist', '')}\n\n"
        f"OUTLINE:\n{outline}\n\nCLAIMS MADE IN THE RECORDING:\n{listed}\n\n"
        f"For each claim, search the web and decide its status as of today:\n"
        f"- confirmed: still accurate today.\n"
        f"- outdated: it was accurate or reasonable when it was said, but something has changed "
        f"since (a newer model or version, a new figure, a price change, a plan that was "
        f"dropped, a prediction that has now resolved).\n"
        f"- incorrect: it was already wrong when it was said.\n"
        f"- disputed: credible sources disagree.\n"
        f"- open: a prediction whose date has not arrived yet. Report the latest evidence for "
        f"and against it. A prediction whose date has passed is confirmed if it came true and "
        f"incorrect if it did not.\n"
        f"- unverified: you could not find good evidence either way.\n\n"
        f"Prefer primary sources (official announcements, documentation, release notes, papers, "
        f"filings) and reputable reporting, and prefer sources dated after the recording was "
        f"published. Treat claims about AI models, products, prices and companies as "
        f"fast-changing and search for the newest information first. Open a page when a search "
        f"snippet is not enough to confirm a figure or a date. A source whose date you cannot "
        f"tell must not decide an 'outdated' verdict on its own. Be specific: name versions, "
        f"numbers and dates.\n\n"
        f"Write one block per claim:\n"
        f"CLAIM n: the claim\nSTATUS: one of the six words\n"
        f"NOW: what is true today, in one or two sentences\n"
        f"AS OF: the date of the newest source you relied on\n"
        f"SOURCES: the URLs you relied on, one per line\n\n"
        f"Then a block DEVELOPMENTS: up to six significant things that happened after the "
        f"recording was published and that a reader of this brief should know, even if no "
        f"claim covers them. For each: a headline, a sentence of detail, the date, the outline "
        f"section number it relates to, and its URLs.",
        max_tokens=MAX_TOKENS["research"])
    sources = _dedupe_sources(sources)
    allowed_urls |= {s["url"] for s in sources}
    log(f"the web check wrote {len(report) // 1000} kB and its tools returned "
        f"{len(sources)} sources")
    if not sources:
        log("WARNING: the web check's tools returned no sources, so no verdict on this page can "
            "be backed by one")
    progress.at("research", 1.0)

    check()
    found = "\n".join(f"{s['url']}" + (f" - {s['title']}" if s.get("title") else "")
                      for s in sources[:80])
    raw = provider.ask_json(
        system,
        f"WEB CHECK REPORT:\n{report}\n\nURLS THE SEARCH TOOLS RETURNED:\n{found or '(none)'}\n\n"
        f"Turn the report into structured verdicts.\n"
        f"- checks: one entry per claim in the report. Keep the claim as the recording stated "
        f"it, its [h:mm:ss] timestamp in 'at', the outline section number it belongs to (0 if "
        f"none), the status the report gave it, 'now' as what is true today, 'as_of' as the "
        f"date, and its sources. Use only URLs from the list above.\n"
        f"- developments: the report's developments, each with its date, the outline section "
        f"number it relates to (0 if none) and its sources from the list above.\n"
        f"- overall: one or two sentences telling the reader how current this recording still "
        f"is, naming the most important change if there is one.",
        schema.VERDICTS_SCHEMA, max_tokens=MAX_TOKENS["verify"], cache_prefix=prefix)
    verdicts = schema.verdicts(raw, allowed_urls, len(plan["sections"]))
    counts = {s: sum(1 for c in verdicts["checks"] if c["status"] == s)
              for s in schema.VERDICT_STATUSES}
    log("web check: " + ", ".join(f"{n} {s}" for s, n in counts.items() if n)
        + f"; {len(verdicts['developments'])} later developments")
    progress.at("verify", 1.0)
    return verdicts, sources


def _section_check(web_check: dict | None, number: int) -> tuple[str, str]:
    """The web check's findings for one section, and the instruction that goes with them."""
    if not web_check:
        return "", ""
    lines = []
    for c in web_check["checks"]:
        if c["section"] == number and c["status"] != "unverified":
            links = " ".join(f"[{s['label']}]({s['url']})" for s in c["sources"][:2])
            lines.append(f"- {c['status'].upper()}: \"{c['claim']}\""
                         + (f" [{c['at']}]" if c["at"] else "")
                         + (f". Now: {c['now']}" if c["now"] else "")
                         + (f" (as of {c['as_of']})" if c["as_of"] else "") + f". {links}")
    for d in web_check["developments"]:
        if d["section"] == number:
            links = " ".join(f"[{s['label']}]({s['url']})" for s in d["sources"][:2])
            lines.append(f"- SINCE THE RECORDING ({d['date'] or 'date unknown'}): "
                         f"{d['headline']}. {d['detail']} {links}")
    if not lines:
        return "", ""
    block = ("\n\nWEB CHECK FOR THIS SECTION (what is true today, found by searching):\n"
             + "\n".join(lines))
    rule = ("\n\nWhere the web check found a claim in this part outdated, incorrect or "
            "disputed, say so where the claim comes up: first what the recording said, then "
            "what is true now and since when, with its link. Keep both; the reader needs to "
            "know what the speaker said as well as what has changed.")
    return block, rule


def _attach_updates(sections: list[dict], web_check: dict) -> None:
    """Put each warning next to the section it concerns, so nobody learns a stale fact."""
    by_number = {s["number"]: s for s in sections}
    for c in web_check["checks"]:
        if c["status"] in schema.VERDICT_WARNINGS and c["section"] in by_number:
            by_number[c["section"]]["updates"].append(c)


def _link_lists(research_sources: list[dict], recording_urls: set[str]) -> str:
    parts = []
    if research_sources:
        parts.append("SOURCES THE WEB CHECK RETURNED:\n" + "\n".join(
            s["url"] + (f" - {s['title']}" if s.get("title") else "")
            for s in research_sources[:60]))
    if recording_urls:
        parts.append("LINKS FROM THE RECORDING AND ITS DESCRIPTION:\n"
                     + "\n".join(sorted(recording_urls)[:40]))
    return ("\n\n" + "\n\n".join(parts)) if parts else ""


def _plan(provider: Provider, system: str, source: str, prefix: str, max_sections: int,
          max_claims: int, check: Callable[[], None], log: Callable[[str], None]) -> dict:
    """The outline pass, retried once with a smaller ask if it comes back with no sections.

    Every other pass can be skipped: a failed section is dropped, a failed web check leaves the
    brief transcript-only. This one cannot, so it is the pass worth a second attempt. The retry
    asks for fewer sections and drops the per-section 'focus' note, which is what a small model
    runs out of room writing, measured on a 100-second clip, where the first attempt was cut off
    part way through a twelve-section plan and left nothing usable behind.
    """
    shape = schema.PLAN_WITH_CLAIMS_SCHEMA if max_claims else schema.PLAN_SCHEMA
    claims_rule = (
        f"\n- claims: up to {max_claims} specific, checkable factual claims from the recording "
        f"that a reader might act on or repeat. Put first the ones most likely to have changed "
        f"since the recording was published: model names and capabilities, benchmark results, "
        f"prices, release dates, company plans, laws and regulations, people's roles, "
        f"statistics, and predictions with a date. For each: the claim in one sentence that "
        f"stands on its own (name the subject, and the time it refers to if the recording "
        f"gave one), its [h:mm:ss] timestamp in 'at', the number of the section it "
        f"belongs to, and whether it is time-sensitive. Skip opinions and anything that cannot "
        f"be checked." if max_claims else "")
    plan: dict = {}
    for attempt, want in enumerate((max_sections, max(MIN_SECTIONS, max_sections // 2))):
        check()
        if attempt:
            log(f"the plan came back without sections; retrying with {want}")
        focus = ("" if attempt else
                 ", a 'focus' note to yourself on exactly what that section must cover and "
                 "what it must leave to other sections")
        plan = schema.plan(provider.ask_json(
            system,
            f"RECORDING:\n{source}\n\n"
            f"Plan a brief on this recording.\n"
            f"- title: a specific headline naming the person or the central claim, not a topic "
            f"label.\n"
            f"- kicker: one line of context: the format, the show, the length, the date.\n"
            f"- subtitle: one sentence on who is talking about what.\n"
            f"- gist: the one-minute version. One paragraph of 50 to 80 words that gives a "
            f"reader who has never heard of this the whole idea: what the recording is, its "
            f"central argument or lesson, and where it lands. Main claim first. Plain words.\n"
            f"- takeaways: three to five key takeaways, each the point in **bold** followed by "
            f"one sentence of at most 20 words. Together with the gist they must read in about "
            f"a minute (around 200 words in all) and leave the reader with the substance, not a "
            f"table of contents.\n"
            f"- why_it_matters: one or two sentences on why this matters to the reader and what "
            f"they could do with it.\n"
            f"- stats: up to five figures actually stated in the recording that a reader would "
            f"want to remember, each with the short label that makes it mean something. Only "
            f"real quantities (amounts, prices, dates, percentages, measured results), never "
            f"counts of the recording's own structure such as its rules, steps, layers or tips. "
            f"Return an empty list rather than padding it. Use tone 'red' for risk, 'green' for "
            f"upside, 'blue' for a forecast, 'violet' for a judgement, 'neutral' otherwise.\n"
            f"- sections: exactly {want} topics, in the order the recording covers them, each "
            f"with a heading, a one-line subheading{focus}, and 'starts_at', the [h:mm:ss] "
            f"timestamp where the topic begins. Cover the whole recording, not just its "
            f"opening.{claims_rule}",
            shape, max_tokens=MAX_TOKENS["plan"], cache_prefix=prefix))
        if plan["sections"]:
            return plan
    return plan


def _try_section(provider: Provider, system: str, prompt: str, *, cache_prefix: str,
                 log: Callable[[str], None], number: int) -> dict:
    """One section fill, where losing this section must not lose the whole page.

    A section can fail on its own: the local model overruns its output window, or the API rate
    limits one call out of twelve. Eleven good sections and a note in the log beat no page at
    all, so the failure is recorded and the loop moves on. `summarize` still refuses to write a
    page with nothing in it.
    """
    try:
        return provider.ask_json(system, prompt, schema.SECTION_SCHEMA,
                                 max_tokens=MAX_TOKENS["sections"], cache_prefix=cache_prefix)
    except SummaryError as e:
        log(f"section {number} failed and was skipped: {e}")
        return {}


def section_budget(spoken_words: int, sections: int) -> int:
    """About how many words each section of the full summary should run to."""
    total = max(SUMMARY_MIN_WORDS, min(SUMMARY_MAX_WORDS, int(spoken_words * SUMMARY_SHARE)))
    return max(SECTION_MIN_WORDS, min(SECTION_MAX_WORDS, total // max(1, sections)))


def _section_count(ceiling: int, duration: float) -> int:
    """How many sections this much recording can actually fill, within `ceiling`."""
    if duration <= 0:
        return max(MIN_SECTIONS, ceiling)
    earned = int(duration // SECONDS_PER_SECTION) + 2
    return max(MIN_SECTIONS, min(ceiling, earned))


def _digest_budget(provider: Provider) -> int:
    """How many characters of notes may go in the cached prefix.

    The prefix is not the whole budget: the question asked alongside it runs to a couple of
    thousand characters, and on the local path the model's window has to hold the answer too.
    """
    return max(1_500, int(provider.context_chars * 0.55))


def _render_notes(note: dict, index: int, total: int) -> str:
    """One excerpt's notes as a compact block, with each field's list capped.

    Capped deliberately hard. Left uncapped, one excerpt's notes reach eight kilobytes and six of
    them overflow a local model's context on their own -- which is how the closing pass came back
    truncated after four sections had written perfectly.
    """
    parts = [f"-- part {index} of {total} --"]
    for key, heading, keep, width in (("topics", "Topics", 4, 90),
                                      ("claims", "Claims", 5, 220),
                                      ("numbers", "Numbers", 4, 90),
                                      ("quotes", "Quotes", 2, 180)):
        items = [i[:width].strip() for i in (note.get(key) or [])[:keep]]
        if items:
            parts.append(heading + ": " + " | ".join(items))
    return "\n".join(parts)


def _digest(notes: list[dict], provider: Provider, system: str, budget: int,
            check: Callable[[], None], log: Callable[[str], None],
            report: Callable[[float], None] | None = None) -> str:
    """Fold per-excerpt notes down to something that fits `budget`, without losing the far end.

    A four-hour recording produces dozens of excerpts, and their notes together are larger than
    any local model's window. The tempting fix is to truncate, and it is the wrong one: the tail
    of the digest is the tail of the recording, so truncating quietly produces a summary of the
    first half labelled as a summary of the whole thing.

    Instead the notes are merged in batches, repeatedly, until they fit. Every round still sees
    every part of the recording, just at lower resolution.
    """
    rendered = [_render_notes(n, i + 1, len(notes)) for i, n in enumerate(notes)]
    text = "\n\n".join(rendered)
    tell = report or (lambda _f: None)
    rounds = 3
    for round_no in range(rounds):
        if len(text) <= budget:
            break
        # Batch size chosen so each merge call's input fits the provider, never fewer than two
        # (a batch of one merges nothing and the loop would never terminate).
        per_batch = max(2, len(rendered) * budget // max(len(text), 1))
        batches = [rendered[i:i + per_batch] for i in range(0, len(rendered), per_batch)]
        log(f"notes are {len(text) // 1000 + 1} kB and the prefix holds {budget // 1000 + 1} kB, "
            f"merging {len(rendered)} parts into {len(batches)} (round {round_no + 1})")
        merged: list[str] = []
        for i, batch in enumerate(batches):
            check()
            note = schema.notes(provider.ask_json(
                system,
                "These are notes from consecutive parts of one recording. Merge them into a "
                "single set of notes, keeping what matters across all of them and dropping "
                "repetition. Keep every concrete number.\n\n" + "\n\n".join(batch),
                schema.NOTES_SCHEMA, max_tokens=MAX_TOKENS["notes"]))
            merged.append(_render_notes(note, i + 1, len(batches)))
            tell((round_no + (i + 1) / len(batches)) / rounds)
        rendered = merged
        text = "\n\n".join(rendered)
    if len(text) > budget:
        # Three rounds of merging and still too big means the provider is returning notes as
        # large as its input. Said out loud rather than silently cut.
        log(f"WARNING: notes still {len(text) // 1000 + 1} kB after merging; the last parts of "
            f"the recording will be under-represented in the summary")
        text = text[:budget]
    tell(1.0)
    return text


class _Cancelled(RuntimeError):
    """Raised inside the passes when the job is cancelled; translated by the pipeline."""


Cancelled = _Cancelled

__all__ = ["summarize", "transcript_text", "Cancelled", "SummaryError", "Unavailable"]
