"""Markdown transcript renderer.

Layout is "timestamped turns + header table": a metadata block, a speaker talk-time summary,
optional LLM summary/chapters, then the body as one bold timestamped heading per speaker turn
with the speech below it, one sentence per line.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from ..merge import Turn
from ..textfmt import hhmmss, hm, hms, human_duration, slugify


def _lang_line(result: dict) -> str:
    lang = result.get("language") or "?"
    prob = result.get("language_probability") or 0.0
    return f"{lang} ({prob:.2f})" if prob else str(lang)


def render(
    turns: list[Turn],
    stats: dict[str, dict],
    result: dict,
    title: str | None = None,
    enrichment: dict | None = None,
) -> str:
    """Build the full Markdown document.

    `result` carries job metadata (source path, duration, model, device, timings).
    `enrichment` is the optional LLM output: {"summary": str, "chapters": [{"start", "title"}]}.
    """
    raw_source = str(result.get("source", "") or "")
    # A downloaded episode's source is a URL. It must NOT go through Path(), which silently
    # collapses "https://" to "https:/" and produces a broken link.
    source_is_url = raw_source.startswith(("http://", "https://"))
    src = Path(raw_source) if not source_is_url else None

    if not title:
        stem = src.stem.replace("_", " ").replace("-", " ").strip() if src else ""
        title = stem or "Transcript"
    duration = float(result.get("duration") or 0.0)
    elapsed = float(result.get("elapsed") or 0.0)
    rtf = (duration / elapsed) if elapsed > 0 else 0.0

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")

    # --- metadata block -------------------------------------------------------------
    n_speakers = len(stats)
    head = f"**Duration** {hms(duration)} · **Speakers** {n_speakers}"
    # Whisper still guesses a language from silence; reporting it on a recording with no speech
    # is just noise ("Language nn (0.61)" on a wordless film).
    if turns:
        head += f" · **Language** {_lang_line(result)}"
    lines.append(head)
    model_bits = [str(result.get("model", "?")),
                  str(result.get("device", "?")).upper(),
                  str(result.get("compute_type", ""))]
    diar = str(result.get("diar_pipeline", "")).split("/")[-1] or "none"
    lines.append(
        f"**Model** {' · '.join(b for b in model_bits if b)} · **Diarization** {diar}"
    )
    uploader = str(result.get("uploader") or "")
    if source_is_url:
        lines.append(f"**Source** [{raw_source}]({raw_source})"
                     + (f" · **Channel** {uploader}" if uploader else ""))
    elif src:
        lines.append(f"**Source** `{src}`")
    stamp = result.get("finished_at") or _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    tail = f"**Transcribed** {stamp}"
    if elapsed:
        tail += f" · {human_duration(elapsed)}"
        if rtf:
            tail += f" ({rtf:.1f}× realtime)"
    lines.append(tail)
    lines.append("")

    # --- speaker table --------------------------------------------------------------
    if stats:
        lines.append("| Speaker | Talk time | Share |")
        lines.append("|---|---|---|")
        for name, s in sorted(stats.items(), key=lambda kv: -kv[1]["seconds"]):
            lines.append(f"| {name} | {hms(s['seconds'])} | {s['share'] * 100:.0f}% |")
        lines.append("")

    # --- optional LLM enrichment ----------------------------------------------------
    enrichment = enrichment or {}
    summary = (enrichment.get("summary") or "").strip()
    if summary:
        lines.append("## Summary")
        lines.append("")
        lines.append(summary)
        lines.append("")

    chapters = enrichment.get("chapters") or []
    if chapters:
        lines.append("## Chapters")
        lines.append("")
        for ch in chapters:
            start, ctitle = float(ch["start"]), str(ch["title"])
            # Link to the anchor planted in the body below, so the chapter list doubles as a
            # table of contents. On a four-hour lecture this is the difference between jumping
            # to 01:47 and scrolling for it.
            anchor = slugify(f"{hm(start)}-{ctitle}")
            lines.append(f"- [**{hm(start)}** {ctitle}](#{anchor})")
        lines.append("")

    lines.append("---")
    lines.append("")

    # --- body -----------------------------------------------------------------------
    chapter_marks = sorted(
        ((float(c["start"]), str(c["title"])) for c in chapters), key=lambda x: x[0]
    )
    ci = 0
    for t in turns:
        while ci < len(chapter_marks) and t.start >= chapter_marks[ci][0]:
            start, ctitle = chapter_marks[ci]
            lines.append(f"<a id=\"{slugify(f'{hm(start)}-{ctitle}')}\"></a>")
            lines.append("")
            ci += 1
        lines.append(f"**[{hhmmss(t.start)}] {t.speaker}**")
        lines.append(t.text)
        lines.append("")

    if not turns:
        lines.append("*No speech detected in this recording.*")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write(path: str | Path, content: str) -> Path:
    """Atomic write — never leave a half-written 4-hour transcript on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
    return path
