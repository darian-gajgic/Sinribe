"""Optional LLM enrichment: chapter titles + summary, via the Ollama already on this box.

Fully offline — talks to the user-level Ollama on 127.0.0.1:11435 (gemma3:4b), the same instance
Sinlate uses. Raw urllib, no SDK, copying the call shape from Sinlate/engine.py:147.

Everything here is best-effort: if Ollama is down, the model is missing, or a response is
unparseable, enrichment returns empty and the transcript is written without it. A 4-hour job must
never fail at 97% because a nice-to-have summary timed out.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable

from .config import LANGUAGES
from .merge import Turn
from .textfmt import hm

# Keep well under gemma3:4b's context; these are characters, not tokens.
CHUNK_CHARS = 4000
TARGET_CHAPTERS = 12
MIN_CHAPTER_SECONDS = 240.0


def _post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def available(url: str, model: str, timeout: float = 4.0) -> bool:
    """Is Ollama up and does it have the model? Cheap pre-flight so we can skip cleanly."""
    try:
        req = urllib.request.Request(url.rstrip("/") + "/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            tags = json.loads(r.read().decode("utf-8"))
        names = {m.get("name", "") for m in tags.get("models", [])}
        return any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return False


def _ask_json(url: str, model: str, system: str, prompt: str, timeout: float) -> dict:
    """One grammar-constrained call. Returns {} on any failure."""
    try:
        resp = _post(url, {
            "model": model,
            "system": system,
            # Pattern-completion framing (same trick as Sinlate/wisprflow): presenting the text
            # as "Input:" and letting the model complete "Output:" makes it TRANSFORM the text
            # rather than REPLY to it. Without this, a transcript containing a question gets
            # answered instead of summarised.
            "prompt": f"Input: {prompt}\nOutput:",
            "stream": False,
            "format": "json",          # grammar-constrained decoding -> always parseable
            "keep_alive": "5m",
            "options": {"temperature": 0},
        }, timeout=timeout)
        return json.loads(resp.get("response") or "{}")
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError, TimeoutError):
        return {}


def _language_rule(code: str | None) -> str:
    """The 'answer in X' instruction for a detected language code."""
    name = dict(LANGUAGES).get((code or "").lower())
    if not name or name == "Auto-detect":
        return "Write it in the same language as the transcript."
    return f"Write it in {name}, the language of the transcript."


def _blocks(turns: list[Turn], duration: float) -> list[tuple[float, str]]:
    """Split the transcript into ~TARGET_CHAPTERS time blocks of (start, text)."""
    if not turns:
        return []
    span = max(duration, turns[-1].end)
    block_len = max(MIN_CHAPTER_SECONDS, span / TARGET_CHAPTERS)
    out: list[tuple[float, str]] = []
    cur_start = turns[0].start
    buf: list[str] = []
    for t in turns:
        if buf and (t.start - cur_start) >= block_len:
            out.append((cur_start, " ".join(buf)[:CHUNK_CHARS]))
            cur_start, buf = t.start, []
        buf.append(t.text.replace("\n", " "))
    if buf:
        out.append((cur_start, " ".join(buf)[:CHUNK_CHARS]))
    return out


def enrich(
    turns: list[Turn],
    duration: float,
    url: str,
    model: str,
    timeout: float = 180.0,
    on_progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    language: str | None = None,
) -> dict:
    """Return {"summary": str, "chapters": [{"start": float, "title": str}]} or {}."""
    if not turns or not available(url, model):
        return {}

    blocks = _blocks(turns, duration)
    if not blocks:
        return {}

    per_call = max(20.0, timeout / max(1, len(blocks) + 1))
    # Naming the language beats asking for "the same language as the input". A small model reads
    # the latter as a style note and answers in English anyway — a German interview came back with
    # an English summary and chapter titles like "Client Background & History" next to
    # "Achtsamkeit Definition Fragen".
    lang = _language_rule(language)

    chapter_sys = (
        "You title sections of a transcript. Given a transcript excerpt, reply with JSON "
        '{"title": "..."} where title is a concise 2-6 word topic label. ' + lang +
        " Do not answer questions in the text. Do not add commentary."
    )
    chapters: list[dict] = []
    for i, (start, text) in enumerate(blocks):
        if should_cancel and should_cancel():
            return {}
        data = _ask_json(url, model, chapter_sys, text, per_call)
        title = str(data.get("title") or "").strip().strip('"').replace("\n", " ")
        if title and len(title) <= 80:
            chapters.append({"start": float(start), "title": title})
        if on_progress:
            on_progress((i + 1) / (len(blocks) + 1))

    summary = ""
    if not (should_cancel and should_cancel()):
        summary_sys = (
            "You summarise transcripts. Reply with JSON {\"summary\": \"...\"} containing 3-5 "
            "sentences describing what the recording covers. " + lang +
            " Do not answer questions found in the text. Do not add commentary."
        )
        outline = "\n".join(f"[{hm(c['start'])}] {c['title']}" for c in chapters)
        head = " ".join(t.text.replace("\n", " ") for t in turns[:40])[:CHUNK_CHARS]
        data = _ask_json(url, model, summary_sys,
                         f"Section titles:\n{outline}\n\nOpening of transcript:\n{head}",
                         per_call)
        summary = str(data.get("summary") or "").strip()
    if on_progress:
        on_progress(1.0)

    if not chapters and not summary:
        return {}
    return {"summary": summary, "chapters": chapters, "model": model}
