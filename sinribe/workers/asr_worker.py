#!/usr/bin/env python3
"""ASR worker — runs under Sinribe's main .venv (CUDA 12 / ctranslate2 stack).

Deliberately a separate PROCESS rather than a thread:
  * VRAM (the weights AND the ~158 MiB CUDA context ctranslate2 pins) is only truly returned
    when the process exits. Nexus learned this the hard way — see app/stt_worker.py:28-32.
  * Cancelling is a kill, with no cooperative-shutdown race.
  * A CUDA OOM here cannot take the GUI down with it.

Protocol: one JSON job object on stdin, newline-delimited JSON events on stdout, human log on
stderr. Never print anything but JSON to stdout.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_OOM_MARKERS = ("out of memory", "cuda_error_out_of_memory", "cublas_status_alloc_failed",
                "failed to allocate", "cudnn_status_alloc_failed")


def emit(**kw) -> None:
    sys.stdout.write(json.dumps(kw, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    sys.stderr.write(f"[asr] {msg}\n")
    sys.stderr.flush()


def is_oom(exc: BaseException) -> bool:
    return any(m in str(exc).lower() for m in _OOM_MARKERS)


def main() -> int:
    raw = sys.stdin.readline()
    if not raw.strip():
        emit(ev="error", msg="no job received on stdin")
        return 2
    job = json.loads(raw)

    wav = job["wav"]
    duration = float(job.get("duration") or 0.0)
    model_name = job.get("model", "large-v3")
    device = job.get("device", "cuda")
    compute_type = job.get("compute_type", "float16")
    batch_size = int(job.get("batch_size", 8))
    beam_size = int(job.get("beam_size", 5))
    patience = float(job.get("patience", 1.0))
    language = job.get("language") or None
    vad_filter = bool(job.get("vad_filter", True))
    sequential = bool(job.get("sequential", False))
    hotwords = (job.get("hotwords") or "").strip() or None
    condition = bool(job.get("condition_on_previous_text", False))
    repetition_penalty = float(job.get("repetition_penalty", 1.0))
    no_repeat_ngram_size = int(job.get("no_repeat_ngram_size", 0))
    prompt_reset_on_temperature = float(job.get("prompt_reset_on_temperature", 0.5))
    temperature_fallback = bool(job.get("temperature_fallback", True))
    vad_min_silence_ms = int(job.get("vad_min_silence_ms", 500))
    vad_speech_pad_ms = int(job.get("vad_speech_pad_ms", 400))

    # HF cache is already populated for large-v3 / medium.en / small; never hit the network.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from faster_whisper import BatchedInferencePipeline, WhisperModel

    t0 = time.time()
    log(f"loading {model_name} on {device} ({compute_type})")
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as e:  # noqa: BLE001
        emit(ev="error", kind="oom" if is_oom(e) else "load", msg=f"{type(e).__name__}: {e}")
        return 1
    log(f"model ready in {time.time() - t0:.1f}s")
    emit(ev="ready", device=device, compute_type=compute_type,
         model=Path(model_name).name if os.path.isdir(model_name) else model_name,
         batch_size=batch_size)

    try:
        kwargs = dict(
            language=language,
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=vad_filter,
        )
        # Proper nouns are where whisper fails most visibly and most fixably: with nothing to
        # expect, "Fujitsu" decodes as *fiuzi* / *jitze* / *future service*. Unlike an
        # initial_prompt — which was tried, and made word accuracy worse by nudging whisper's
        # whole speaking style — hotwords bias only the vocabulary.
        if hotwords:
            kwargs["hotwords"] = hotwords
        if not temperature_fallback:
            # A single temperature means every window keeps its beam search. Whisper's own
            # repeat guard is what the fallback ladder normally provides, so this is only safe
            # with the compression-ratio check still armed (it is, by default).
            kwargs["temperature"] = [0.0]
        if sequential:
            # Tighter VAD than the default (2000 ms silence / 400 ms pad): long silences were
            # swallowing the quiet run-in of the next utterance. Measured on a 34-minute German
            # interview, this plus sequential decoding raises the share of audio covered by word
            # spans from 0.75 to 0.79.
            if vad_filter:
                kwargs["vad_parameters"] = dict(min_silence_duration_ms=vad_min_silence_ms,
                                                speech_pad_ms=vad_speech_pad_ms)
            if beam_size > 1 and patience != 1.0:
                kwargs["patience"] = patience
            # Feeding the previous window back as context is what lets whisper resolve the
            # function words that dominate its errors on fast speech ("das"/"es", "dann"/"da").
            # Left unguarded it also lets a long recording drift into degenerate repeats
            # ("...mehr... ...mehr..."), which is why it used to be off entirely. The repetition
            # guards below are the supported fix for that failure, so context can stay on:
            # prompt_reset_on_temperature drops the context whenever decoding falls back to a
            # higher temperature, i.e. exactly when the model has started to lose the thread.
            kwargs["condition_on_previous_text"] = condition
            if condition:
                kwargs["prompt_reset_on_temperature"] = prompt_reset_on_temperature
            if repetition_penalty != 1.0:
                kwargs["repetition_penalty"] = repetition_penalty
            if no_repeat_ngram_size:
                kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        if batch_size > 1 and not sequential:
            segments, info = BatchedInferencePipeline(model=model).transcribe(
                wav, batch_size=batch_size, **kwargs)
        else:
            segments, info = model.transcribe(wav, **kwargs)

        total = duration or float(getattr(info, "duration", 0.0) or 0.0)
        emit(ev="info",
             language=getattr(info, "language", None),
             language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
             duration=total)

        n_seg = 0
        n_words = 0
        last_report = 0.0
        for seg in segments:
            words = []
            for w in (seg.words or []):
                if w.start is None or w.end is None:
                    continue
                words.append({"start": round(float(w.start), 3),
                              "end": round(float(w.end), 3),
                              "word": w.word,
                              "prob": round(float(w.probability or 0.0), 4)})
            # A segment with word_timestamps off or dropped words still contributes its text,
            # spread across the segment span, so nothing vanishes from the transcript.
            if not words and (seg.text or "").strip():
                words = [{"start": round(float(seg.start), 3), "end": round(float(seg.end), 3),
                          "word": seg.text.strip(), "prob": 0.0}]
            n_seg += 1
            n_words += len(words)
            emit(ev="segment", start=round(float(seg.start), 3), end=round(float(seg.end), 3),
                 text=seg.text, words=words)
            now = float(seg.end)
            if total > 0 and now - last_report >= 1.0:
                last_report = now
                emit(ev="progress", done=min(now, total), total=total)

        elapsed = time.time() - t0
        rtf = (total / elapsed) if elapsed > 0 else 0.0
        log(f"done: {n_seg} segments, {n_words} words in {elapsed:.1f}s ({rtf:.1f}x realtime)")
        emit(ev="done", elapsed=round(elapsed, 2), segments=n_seg, words=n_words,
             realtime_factor=round(rtf, 2))
        return 0
    except Exception as e:  # noqa: BLE001
        emit(ev="error", kind="oom" if is_oom(e) else "runtime",
             msg=f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
