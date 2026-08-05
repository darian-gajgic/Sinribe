"""SRT and WebVTT export with speaker prefixes.

Turns are split into short cues: subtitle lines longer than ~2 lines of text are unreadable, so
each turn is chunked on word boundaries to a character budget and a maximum on-screen duration.
"""

from __future__ import annotations

from pathlib import Path

from ..merge import Turn
from ..textfmt import srt_time, vtt_time

MAX_CHARS = 84       # ~2 lines of 42 characters
MAX_SECONDS = 6.0
MIN_SECONDS = 0.7


def _cues(turns: list[Turn]) -> list[tuple[float, float, str, str]]:
    """Split turns into (start, end, speaker, text) cues."""
    out: list[tuple[float, float, str, str]] = []
    for t in turns:
        words = [w for w in t.words if (w.word or "").strip()]
        if not words:
            out.append((t.start, max(t.end, t.start + MIN_SECONDS), t.speaker,
                        " ".join(t.text.split())))
            continue
        cur: list = []
        chars = 0
        for w in words:
            too_long = chars + len(w.word) > MAX_CHARS
            too_slow = cur and (w.end - cur[0].start) > MAX_SECONDS
            if cur and (too_long or too_slow):
                text = " ".join(x.word.strip() for x in cur).strip()
                out.append((cur[0].start, max(cur[-1].end, cur[0].start + MIN_SECONDS),
                            t.speaker, text))
                cur, chars = [], 0
            cur.append(w)
            chars += len(w.word) + 1
        if cur:
            text = " ".join(x.word.strip() for x in cur).strip()
            out.append((cur[0].start, max(cur[-1].end, cur[0].start + MIN_SECONDS),
                        t.speaker, text))
    return out


def render_srt(turns: list[Turn]) -> str:
    parts = []
    for i, (start, end, speaker, text) in enumerate(_cues(turns), 1):
        parts.append(f"{i}\n{srt_time(start)} --> {srt_time(end)}\n[{speaker}] {text}\n")
    return "\n".join(parts)


def render_vtt(turns: list[Turn]) -> str:
    parts = ["WEBVTT", ""]
    for start, end, speaker, text in _cues(turns):
        parts.append(f"{vtt_time(start)} --> {vtt_time(end)}")
        parts.append(f"<v {speaker}>{text}")
        parts.append("")
    return "\n".join(parts)


def write(path: str | Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
    return path
