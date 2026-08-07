"""Sidecar JSON: the complete machine-readable result next to the .md.

This is the insurance policy for multi-hour jobs. Because it stores word-level timestamps and the
raw diarization, renaming a speaker or re-exporting to a different format re-renders in under a
second instead of re-running a 20-minute GPU job.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..merge import DiarTurn, Turn, Word, speaker_stats

SCHEMA_VERSION = 1


def build(
    turns: list[Turn],
    stats: dict[str, dict],
    result: dict,
    diar_turns: list[DiarTurn] | None = None,
    enrichment: dict | None = None,
    settings: dict | None = None,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        # `speed_target`, `decode` and `speech_coverage` are here so a transcript records how it
        # was produced and how completely — which is what makes two runs of the same recording
        # comparable after the fact, and what `sinribe-eval` reads when scoring a sweep.
        "meta": {k: result.get(k) for k in (
            "source", "duration", "language", "language_probability", "model", "device",
            "compute_type", "batch_size", "diar_pipeline", "elapsed", "finished_at",
            "realtime_factor", "title", "speed_target", "decode", "speech_coverage",
            "passes", "vote_agreement")},
        "stats": {k: {kk: vv for kk, vv in v.items() if kk != "longest"} | {
            "longest": list(v["longest"])} for k, v in stats.items()},
        "speaker_names": {t.speaker: t.speaker for t in turns},
        "turns": [t.to_dict() for t in turns],
        "diar_turns": [{"start": d.start, "end": d.end, "speaker": d.speaker}
                       for d in (diar_turns or [])],
        "enrichment": enrichment or {},
        "settings": settings or {},
    }


def write(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def load(path: str | Path) -> tuple[list[Turn], dict, dict, dict]:
    """Read a sidecar back into (turns, stats, meta, payload) for re-rendering."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    turns = []
    for t in payload.get("turns", []):
        turns.append(Turn(
            speaker=t["speaker"], raw=t.get("raw", t["speaker"]),
            start=float(t["start"]), end=float(t["end"]), text=t["text"],
            words=[Word(start=float(w["start"]), end=float(w["end"]), word=w["word"],
                        prob=float(w.get("prob", 0.0)), speaker=w.get("speaker"))
                   for w in t.get("words", [])],
        ))
    return turns, speaker_stats(turns), payload.get("meta", {}), payload


def apply_names(turns: list[Turn], names: dict[str, str]) -> list[Turn]:
    """Rename display speakers in place, e.g. {'Person 1': 'Prof. Müller'}."""
    for t in turns:
        if t.speaker in names and names[t.speaker].strip():
            t.speaker = names[t.speaker].strip()
    return turns
