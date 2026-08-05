#!/usr/bin/env python3
"""Speaker-refinement worker — runs under .venv-diar (CUDA 13 / torch stack).

Diarization decides *when* the speaker changes, from segmentation and clustering over the whole
recording. It is good at that and bad at short spans: a two-second answer wedged between two long
questions is routinely swallowed by the neighbouring turn, and no downstream regrouping can undo
it, because by then the wrong label is all there is.

This stage asks a different, much easier question, once per *sentence*: which of these voices does
this particular audio sound like? It builds one voice print per speaker from the diarization's
long, unambiguous turns, embeds each sentence's own audio, and reports the nearest speaker plus
how decisive the match was. `merge.vote_sentences` overrides the diarization only where the
verdict is decisive, so a confident correction is applied and a coin-flip on "Okay." is not.

Only the embedding model is loaded (~2 s, a few hundred MiB) — the segmentation and clustering
halves of the diarization pipeline are not needed and are not touched.

Protocol: one JSON job object on stdin, newline-delimited JSON events on stdout, log on stderr.
"""

from __future__ import annotations

import json
import os
import sys
import time

DEFAULT_PIPELINE = "pyannote/speaker-diarization-community-1"

# A turn contributes to a voice print only if it is at least this long and stands this far clear
# of any differently-labelled neighbour, so cross-talk at a boundary cannot poison the reference.
TRUSTED_MIN_S = 2.0
TRUSTED_ISOLATION_S = 0.5
# One long monologue must not dominate its speaker's voice print.
TRUSTED_CAP_S = 10.0
# Below this a crop carries too little voice to embed at all.
MIN_EMBED_S = 0.35


def emit(**kw) -> None:
    sys.stdout.write(json.dumps(kw, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    sys.stderr.write(f"[refine] {msg}\n")
    sys.stderr.flush()


def trusted_turns(turns: list[dict]) -> list[dict]:
    ordered = sorted(turns, key=lambda t: (t["start"], t["end"]))
    out = []
    for i, t in enumerate(ordered):
        if t["end"] - t["start"] < TRUSTED_MIN_S:
            continue
        prev_ok = (i == 0 or ordered[i - 1]["speaker"] == t["speaker"]
                   or t["start"] - ordered[i - 1]["end"] >= TRUSTED_ISOLATION_S)
        next_ok = (i == len(ordered) - 1 or ordered[i + 1]["speaker"] == t["speaker"]
                   or ordered[i + 1]["start"] - t["end"] >= TRUSTED_ISOLATION_S)
        if prev_ok and next_ok:
            out.append(t)
    return out


def main() -> int:  # noqa: C901 - a linear worker; splitting it would only scatter the protocol
    raw = sys.stdin.readline()
    if not raw.strip():
        emit(ev="error", msg="no job received on stdin")
        return 2
    job = json.loads(raw)

    wav = job["wav"]
    sentences = job.get("sentences") or []
    turns = job.get("turns") or []
    pipeline_name = job.get("pipeline") or DEFAULT_PIPELINE
    token = job.get("hf_token") or None
    device_pref = job.get("device", "cuda")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    speakers = sorted({t["speaker"] for t in turns})
    if len(speakers) < 2 or not sentences:
        # Nothing to disambiguate: one voice, or no text. Say so and let the merge fall through
        # to the plain diarization result rather than paying for a model load.
        log(f"skipped: {len(speakers)} speaker(s), {len(sentences)} sentence(s)")
        emit(ev="result", scores=[])
        emit(ev="done", elapsed=0.0, scored=0, speakers=len(speakers))
        return 0

    t0 = time.time()
    import numpy as np
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        import soundfile as sf
        from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    except Exception as e:  # noqa: BLE001
        emit(ev="error", kind="load", msg=f"{type(e).__name__}: {e}")
        return 1

    device = "cuda" if device_pref == "cuda" and torch.cuda.is_available() else "cpu"
    try:
        model = PretrainedSpeakerEmbedding(
            {"checkpoint": pipeline_name, "subfolder": "embedding"},
            device=torch.device(device), use_auth_token=token)
    except TypeError:
        # Older/newer signatures disagree on the token kwarg; the model is cached anyway.
        try:
            model = PretrainedSpeakerEmbedding(
                {"checkpoint": pipeline_name, "subfolder": "embedding"},
                device=torch.device(device))
        except Exception as e:  # noqa: BLE001
            emit(ev="error", kind="load", msg=f"{type(e).__name__}: {e}")
            return 1
    except Exception as e:  # noqa: BLE001
        emit(ev="error", kind="load", msg=f"{type(e).__name__}: {e}")
        return 1

    log(f"embedding model on {device} in {time.time() - t0:.1f}s")
    emit(ev="ready", device=device, speakers=len(speakers))

    try:
        data, sr = sf.read(wav, dtype="float32", always_2d=True)
        audio = torch.from_numpy(data.T)  # (channel, time)
        n_samples = audio.shape[1]

        def embed(start: float, end: float, cap: float | None = None) -> "np.ndarray | None":
            a = max(0, int(start * sr))
            b = min(n_samples, int((min(end, start + cap) if cap else end) * sr))
            if b - a < int(MIN_EMBED_S * sr):
                return None
            chunk = audio[:, a:b].unsqueeze(0).to(device)
            with torch.no_grad():
                v = model(chunk)
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(v)):
                return None
            n = float(np.linalg.norm(v))
            return v / n if n > 0 else None

        picked = trusted_turns(turns)
        acc: dict[str, list] = {}
        for t in picked:
            v = embed(t["start"], t["end"], cap=TRUSTED_CAP_S)
            if v is not None:
                acc.setdefault(t["speaker"], []).append(v)

        centroids: dict[str, "np.ndarray"] = {}
        for spk, vs in acc.items():
            m = np.mean(np.stack(vs), axis=0)
            n = float(np.linalg.norm(m))
            if n > 0:
                centroids[spk] = m / n
        log(f"voice prints from {len(picked)} trusted turns: "
            + ", ".join(f"{s}={len(acc.get(s, []))}" for s in speakers))

        if len(centroids) < 2:
            # Not enough clean audio to characterise both voices; refining on one voice print
            # would just re-assert the diarization. Return nothing and change no decisions.
            log("not enough isolated speech to build two voice prints — leaving diarization as is")
            emit(ev="result", scores=[])
            emit(ev="done", elapsed=round(time.time() - t0, 2), scored=0,
                 speakers=len(centroids))
            return 0

        keys = sorted(centroids)
        sep = float(centroids[keys[0]] @ centroids[keys[1]]) if len(keys) == 2 else 0.0
        log(f"voice-print separation (cosine, lower is better): {sep:.3f}")

        scores = []
        scored = 0
        last = 0.0
        for n_i, s in enumerate(sentences):
            v = embed(float(s["start"]), float(s["end"]))
            if v is None:
                scores.append({"speaker": None, "margin": 0.0})
            else:
                sims = sorted(((float(v @ c), spk) for spk, c in centroids.items()), reverse=True)
                margin = sims[0][0] - sims[1][0] if len(sims) > 1 else 0.0
                scores.append({"speaker": sims[0][1], "margin": round(margin, 4)})
                scored += 1
            now = time.time()
            if now - last >= 0.4:
                last = now
                emit(ev="progress", done=(n_i + 1) / len(sentences), total=1.0)

        elapsed = time.time() - t0
        log(f"done: scored {scored}/{len(sentences)} sentences in {elapsed:.1f}s")
        emit(ev="result", scores=scores, separation=round(sep, 4))
        emit(ev="done", elapsed=round(elapsed, 2), scored=scored, speakers=len(centroids))
        return 0
    except Exception as e:  # noqa: BLE001
        emit(ev="error", kind="runtime", msg=f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
