"""Markdown transcript renderer.

Layout is "timestamped turns + header table": a metadata block, a speaker talk-time summary,
optional LLM summary/chapters, then the body as one bold timestamped heading per speaker turn
with the speech below it, one sentence per line.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from .. import presets
from ..merge import Turn
from ..textfmt import hhmmss, hm, hms, human_duration, slugify


def _lang_line(result: dict) -> str:
    lang = result.get("language") or "?"
    prob = result.get("language_probability") or 0.0
    return f"{lang} ({prob:.2f})" if prob else str(lang)


def _join(words: list) -> str:
    """Join words back into text.

    Whisper emits each word with its leading space, so concatenation is usually right — but a
    word list that has been through a rename, an edit or a round trip may not carry them, and
    concatenating those produces "w4w5w6". Only insert a space where neither side has one.
    """
    out = ""
    for w in words:
        piece = w.word
        if out and not piece[:1].isspace() and not out[-1:].isspace():
            out += " "
        out += piece
    return out.strip()


def review_spans(
    turns: list[Turn],
    threshold: float = 0.5,
    context: int = 4,
    max_spans: int = 0,
) -> list[tuple[float, str, str]]:
    """Passages the recogniser was unsure of, as (start, speaker, quoted text with context).

    Whisper's confidence is a weak predictor of error in general — measured on a hard German
    interview, most wrong words came back with a probability around 0.86, which is why this is a
    review aid and not a correction mechanism. What it is good at is the bottom of the range:
    below 0.5 sits about 5 % of the transcript, and 57 % of those words are genuinely wrong.

    Runs are merged across a couple of confident words so that "…, [gap], …" reads as one
    passage rather than three, which is how someone correcting a transcript would hear it.

    `max_spans` 0 means no limit, which is the default: the list is a proofreading checklist, and
    a checklist that silently stops two thirds of the way through is worse than a long one. The
    benchmark interview produces 363 entries over 65 minutes — long, but it is the complete set
    of places worth an ear, and working through it beats re-listening to the whole recording.

    A positive `max_spans` caps the list, keeping the LEAST confident entries and then restoring
    time order for reading. Truncating in time order instead would quietly hand back a review list
    covering only the first ten minutes.
    """
    scored: list[tuple[float, float, str, str]] = []
    for t in turns:
        words = t.words or []
        i = 0
        while i < len(words):
            if words[i].prob >= threshold:
                i += 1
                continue
            start = i
            end = i
            gap = 0
            for j in range(i, len(words)):
                if words[j].prob < threshold:
                    end, gap = j, 0
                elif gap < 2:
                    gap += 1
                else:
                    break
            lo, hi = max(0, start - context), min(len(words), end + context + 1)
            marked = _join(words[start:end + 1])
            if marked:
                # Assembled from the three slices rather than by substituting into the joined
                # text: the uncertain words often recur in their own context ("das das"), and a
                # substitution would then emphasise the wrong occurrence.
                pieces = [_join(words[lo:start]), f"**{marked}**", _join(words[end + 1:hi])]
                doubt = min(w.prob for w in words[start:end + 1])
                scored.append((doubt, words[start].start, t.speaker,
                               " ".join(p for p in pieces if p)))
            i = end + 1

    scored.sort(key=lambda s: s[0])          # least confident first
    keep = scored[:max_spans] if max_spans > 0 else scored
    keep.sort(key=lambda s: s[1])            # ... then back into reading order
    return [(start, speaker, text) for _, start, speaker, text in keep]


def count_review_spans(turns: list[Turn], threshold: float = 0.5) -> int:
    """How many passages fall below `threshold` in total, before any cap is applied."""
    return len(review_spans(turns, threshold=threshold, max_spans=0))


def render(
    turns: list[Turn],
    stats: dict[str, dict],
    result: dict,
    title: str | None = None,
    enrichment: dict | None = None,
    review: bool = False,
    review_max: int = 0,
) -> str:
    """Build the full Markdown document.

    `result` carries job metadata (source path, duration, model, device, timings).
    `enrichment` is the optional LLM output: {"summary": str, "chapters": [{"start", "title"}]}.
    `review` appends the list of passages worth checking by ear (see review_spans);
    `review_max` caps that list, 0 meaning all of them.
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
    model_line = f"**Model** {' · '.join(b for b in model_bits if b)} · **Diarization** {diar}"
    # Recording the quality setting makes two transcripts of the same recording comparable
    # afterwards, which otherwise means remembering where the slider was.
    if result.get("speed_target"):
        model_line += f" · **Quality** {result['speed_target']}× target"
    # The pass count is the part of the quality setting that actually cost the time, so it is
    # worth naming separately rather than leaving it implied by the rung.
    if int(result.get("passes") or 1) > 1:
        model_line += f" · **Voted** {result['passes']} passes"
        if result.get("vote_agreement") is not None:
            model_line += f" ({float(result['vote_agreement']):.0%} unanimous)"
    lines.append(model_line)
    uploader = str(result.get("uploader") or "")
    if source_is_url:
        published = str(result.get("published") or "")
        lines.append(f"**Source** [{raw_source}]({raw_source})"
                     + (f" · **Channel** {uploader}" if uploader else "")
                     + (f" · **Published** {published}" if published else ""))
    elif src:
        lines.append(f"**Source** `{src}`")
    stamp = result.get("finished_at") or _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    tail = f"**Transcribed** {stamp}"
    if elapsed:
        tail += f" · {human_duration(elapsed)}"
        # Not the end-to-end factor: on a resumed job that measures the checkpoint cache, and a
        # transcript that claims 29.5x for a 10.7x setting is a lie its own reader cannot catch.
        note = presets.speed_note(result)
        tail += note or (f" ({rtf:.1f}× realtime)" if rtf else "")
    lines.append(tail)
    # Loud, and in the document rather than only in a log line that scrolls away: a truncated
    # transcript otherwise looks exactly like a short recording.
    coverage = result.get("speech_coverage")
    if coverage is not None and float(coverage) < 0.5 and turns:
        lines.append("")
        lines.append(f"> ⚠️ **Only {float(coverage):.0%} of the detected speech was transcribed.** "
                     f"This transcript is very likely truncated — try a different quality "
                     f"setting or model before relying on it.")
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

    # --- optional review list -------------------------------------------------------
    if review and turns:
        spans = review_spans(turns, max_spans=review_max)
        if spans:
            total = count_review_spans(turns)
            lines.append("---")
            lines.append("")
            lines.append("## Worth a listen")
            lines.append("")
            intro = ("Passages the recogniser was least sure of, bold where it wavered. "
                     "Rather more than half of these are genuinely wrong, so this is where "
                     "checking by ear pays; the rest of the transcript is a much safer bet.")
            if total > len(spans):
                intro += (f" Showing the {len(spans)} least confident of {total} — a recording "
                          f"this difficult has more doubt in it than one sitting can cover.")
            else:
                intro += (f" All {total} of them are here, in time order, so this doubles as a "
                          f"checklist to work through against the audio.")
            lines.append(intro)
            lines.append("")
            for start, speaker, text in spans:
                lines.append(f"- **[{hhmmss(start)}]** {speaker}: {text}")
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
