"""Deterministic text formatting helpers — no LLM involved.

`format_sentences` is a port of format_notes() from local-wisprflow/wf_daemon.py:244,
including its curated EN+DE abbreviation set. Whisper large-v3 already punctuates well, so
splitting on punctuation gives one sentence per line, which is what the transcript body wants.
"""

from __future__ import annotations

import re

# Only UNAMBIGUOUS abbreviations. Words like "no", "st", "co", "al", "ca" are deliberately absent
# because they are also ordinary sentence-ending words ("I said no.") and would block real splits.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "eg", "ie",
    "e.g", "i.e", "a.m", "p.m", "u.s", "u.k", "nr", "vol", "fig", "inc",
    "ltd", "corp", "dept", "approx", "cf", "gov", "sen",
    # German
    "z.b", "d.h", "u.a", "u.s.w", "usw", "bzw", "ggf", "evtl", "bspw", "sog",
}

# A run of sentence-ending punctuation, optional closing quote/bracket, then whitespace.
_SENT_BOUNDARY = re.compile(r"[.!?…]+[\"')\]”’]*\s+(?=\S)")

# Same punctuation run, anchored at the end of a single token.
_SENT_END = re.compile(r"[.!?…]+[\"')\]”’]*$")


def is_sentence_end(token: str, alone: bool = False) -> bool:
    """True if `token` closes a sentence, applying format_sentences' abbreviation guards.

    Exposed so that callers splitting a *word list* (merge.py, which needs sentence boundaries as
    word indices rather than as string offsets) apply exactly the same rule as the renderer —
    otherwise a sentence could be grouped one way for speaker attribution and printed another.

    `alone` says the token would be the whole sentence, which is what distinguishes a standalone
    list marker ("1.") from a clause that merely ends in a number ("I scored 8.").
    """
    token = (token or "").strip()
    if not token or not _SENT_END.search(token):
        return False
    bare = _SENT_END.sub("", token).lower().rstrip(".")
    if bare in _ABBREV:
        return False
    if len(bare) == 1 and bare.isalpha():   # initial, e.g. "J." in "J. R. R."
        return False
    if alone and bare.isdigit():            # standalone list marker
        return False
    return True


def format_sentences(text: str) -> str:
    """Return `text` with each sentence on its own line.

    A break after an abbreviation ("Dr.", "e.g."), a single-letter initial ("A."), or a
    STANDALONE list marker ("1.") is suppressed to avoid choppy output — but a clause that
    merely ends in a number ("I scored 8.") still splits.
    """
    text = " ".join((text or "").split())  # normalise all whitespace/newlines to single spaces
    if not text:
        return text
    lines, i = [], 0
    for m in _SENT_BOUNDARY.finditer(text):
        prev = text[i:m.start()]
        words = prev.split()
        last = words[-1].lower().rstrip(".") if words else ""
        if (last in _ABBREV
                or (len(last) == 1 and last.isalpha())      # initial, e.g. "J." in "J. R. R."
                or (len(words) == 1 and last.isdigit())):   # standalone list marker, e.g. "1."
            continue                        # not a real sentence end — keep building this line
        lines.append(text[i:m.start()] + m.group().strip())  # sentence + its punctuation
        i = m.end()                         # skip the whitespace after the boundary
    tail = text[i:].strip()
    if tail:
        lines.append(tail)
    return "\n".join(s.strip() for s in lines if s.strip())


def hms(seconds: float) -> str:
    """Seconds -> H:MM:SS, always with hours (transcripts here run to 4 h)."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def hhmmss(seconds: float) -> str:
    """Seconds -> HH:MM:SS zero-padded, for timestamp markers in the body."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def hm(seconds: float) -> str:
    """Seconds -> H:MM, for chapter headings."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, _ = divmod(rem, 60)
    return f"{h:02d}:{m:02d}"


def human_duration(seconds: float) -> str:
    """Seconds -> '0.4s' / '45s' / '9m 12s' / '1h 04m', for elapsed/ETA readouts."""
    seconds = max(0.0, float(seconds))
    if seconds < 10:
        # A job resumed entirely from cache finishes in well under a second; rounding that to
        # a bare "0s" reads like nothing happened.
        return f"{seconds:.1f}s"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def srt_time(seconds: float) -> str:
    """Seconds -> 'HH:MM:SS,mmm' (SubRip)."""
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vtt_time(seconds: float) -> str:
    """Seconds -> 'HH:MM:SS.mmm' (WebVTT)."""
    return srt_time(seconds).replace(",", ".")


def slugify(text: str) -> str:
    """GitHub-style anchor slug, used for the chapter table of contents."""
    s = re.sub(r"[^\w\s-]", "", (text or "").lower(), flags=re.UNICODE)
    s = re.sub(r"[\s_]+", "-", s).strip("-")
    return s or "section"
