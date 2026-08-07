"""Decode one file several ways and put the scores side by side.

This is the loop the whole accuracy effort runs on: change one knob, re-decode, score, keep it
only if the number moved. It is cheap because the expensive stages are already checkpointed —
`decoded.wav` and `diar.json` in ~/.cache/sinribe/jobs/<hash>/ do not depend on decode settings,
so the second variant onwards pays for the ASR pass alone.

A variants file is JSON:

    {"variants": [
       {"name": "baseline",       "asr_model": "large-v3",        "speed_target": 6,
        "decode_overrides": {"condition_on_previous_text": false}},
       {"name": "german+context", "asr_model": "large-v3-german", "speed_target": 6},
       {"name": "german+hotwords","asr_model": "large-v3-german", "speed_target": 6,
        "hotwords": "Fujitsu, Siemens, VR-Bank"}
    ]}

Any key is merged over the saved config; `decode_overrides` reaches the knobs in presets.py
without every experiment needing its own config key.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import DEFAULTS
from .normalize import Normalizer
from .reference import load_hypothesis, load_reference
from .score import Report, score


def _run_variant(audio: Path, out_dir: Path,
                 variant: dict) -> tuple[Path, float, dict, list[str]]:
    """Transcribe `audio` with one variant's settings.

    Returns (markdown, seconds, run info, stages that had to be recomputed). That last one is not
    decoration: a variant whose diarization checkpoint was cold pays roughly two extra minutes
    that have nothing to do with the setting under test, and an unflagged run like that already
    cost an hour of chasing a 197-second result that was really 77 seconds of transcription.
    """
    from ..pipeline import JobSpec, Runner, _job_key
    from ..config import JOBS_DIR

    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in variant.items() if k != "name"})
    # A sweep measures the recogniser, so the trimmings are off: chapter/summary generation adds
    # minutes of LLM time and subtitle files nobody scores.
    cfg.update({"llm_enrich": False, "write_srt": False, "write_vtt": False,
                "write_json": True, "output_dir": str(out_dir),
                # Keep the decoded WAV between variants. Normal runs delete it; a sweep would
                # then pay for the same ffmpeg pass once per variant.
                "keep_decoded_wav": True})

    cache = JOBS_DIR / _job_key(audio)
    cold = [name for name, f in (("decode", "decoded.wav"), ("diarize", "diar.json"))
            if not (cache / f).exists()]
    # A voting rung's passes are checkpointed individually, so a variant that shares passes with
    # an earlier one runs only the new ones and reports a wildly optimistic speed. Caught the
    # first time it happened: a three-pass rung measured 42x because two passes were already on
    # disk, against a true 10.7x.
    warm_passes = len(list(cache.glob("asr-pass*.json")))
    if warm_passes:
        cold.append(f"{warm_passes} cached pass(es) — speed flattered")

    spec = JobSpec(input_path=audio, output_dir=out_dir, cfg=cfg)
    runner = Runner(spec, on_log=lambda m: None)
    t0 = time.time()
    result = runner.run()
    return Path(result["markdown_path"]), time.time() - t0, result, cold


def _table(rows: list[tuple[str, Report, float, float, list[str]]]) -> str:
    """Comparison table, best WER first."""
    head = (f"{'variant':<28} {'WER':>7} {'correct':>8} {'sub':>6} {'del':>6} {'ins':>6} "
            f"{'spk':>6} {'RTF':>7}")
    lines = [head, "-" * len(head)]
    best = min((r.total.wer for _, r, _, _, _ in rows), default=0.0)
    for name, rep, _elapsed, rtf, cold in sorted(rows, key=lambda x: x[1].total.wer):
        t = rep.total
        mark = "  <-- best" if t.wer == best and len(rows) > 1 else ""
        # A cold run's RTF describes the cache, not the setting. Marked rather than dropped,
        # because its WER is still perfectly good.
        lines.append(f"{name:<28} {t.wer:>6.1%} {t.accuracy:>7.1%} {t.sub:>6} {t.dele:>6} "
                     f"{t.ins:>6} {rep.speaker_accuracy:>5.1%} "
                     f"{rtf:>6.1f}x{'⚠' if cold else ' '}{mark}")
    if any(cold for *_, cold in rows):
        lines.append("")
        lines.append("⚠ timing includes a stage that had to be recomputed — compare its WER, "
                     "not its speed.")
    return "\n".join(lines)


def run_sweep(configs: Path, audio: Path, reference: Path,
              out: Path | None = None, keep_fillers: bool = False) -> int:
    """Decode `audio` once per variant, score each against `reference`, print the comparison."""
    spec = json.loads(Path(configs).read_text(encoding="utf-8"))
    variants = spec.get("variants") or []
    if not variants:
        print(f"ERROR: no variants in {configs}")
        return 2

    work = Path(spec.get("output_dir") or (Path.home() / ".cache" / "sinribe" / "sweep"))
    work.mkdir(parents=True, exist_ok=True)
    nz = Normalizer(drop_fillers=not keep_fillers)
    ref = load_reference(reference)
    print(f"reference {reference.name}: {len(ref)} turns\n")

    rows: list[tuple[str, Report, float, float]] = []
    results = []
    for i, variant in enumerate(variants, 1):
        name = str(variant.get("name") or f"variant{i}")
        print(f"[{i}/{len(variants)}] {name} ...", flush=True)
        try:
            md, elapsed, info, cold = _run_variant(audio, work / name, variant)
        except Exception as e:  # noqa: BLE001 - one bad variant must not lose the whole sweep
            print(f"    FAILED: {type(e).__name__}: {e}")
            results.append({"name": name, "error": f"{type(e).__name__}: {e}"})
            continue
        rep = score(ref, load_hypothesis(md), nz, label=name)
        rtf = float(info.get("duration", 0.0)) / elapsed if elapsed else 0.0
        rows.append((name, rep, elapsed, rtf, cold))
        warn = f"   ⚠ timing includes {' + '.join(cold)}" if cold else ""
        print(f"    WER {rep.total.wer:.1%}   {rtf:.1f}x realtime   ({elapsed:.0f}s){warn}",
              flush=True)
        results.append({**rep.to_dict(), "name": name, "seconds": round(elapsed, 1),
                        "realtime_factor": round(rtf, 2), "cold_stages": cold,
                        "settings": variant})

    print("\n" + _table(rows))
    print("\nRTF is end-to-end wall clock. The first variant of a sweep usually looks slower "
          "than it is:\nit pays for the shared decode and diarization that every later variant "
          "reuses. Anything that\nchanges the audio (audio_filter) pays for them again.")
    if out:
        Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote {out}")
    return 0
