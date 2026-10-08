"""Job orchestrator: decode -> diarize -> transcribe -> merge -> enrich -> render.

Runs entirely off the GUI thread. Both GPU stages are separate subprocesses in separate venvs,
executed sequentially, so peak VRAM is max(diarize, transcribe) rather than their sum and the
CUDA context is genuinely returned to the driver between them.
"""

from __future__ import annotations

import hashlib
import json
import re
import os
import subprocess
import threading
import time
import datetime as _dt
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from . import audio, enrich as enrich_mod, fetch, presets, rover, summary as summary_mod
from .config import (
    DOWNLOADS_DIR, JOBS_DIR, hf_token, model_available, resolve_model, supports_hotwords,
)
from .merge import DiarTurn, SentenceScore, Word, merge, sentence_spans
from .render import markdown, sidecar, subtitles, webpage
from .textfmt import human_duration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASR_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
DIAR_PYTHON = PROJECT_ROOT / ".venv-diar" / "bin" / "python"

# Relative shares of the overall progress bar. Transcription dominates; diarization is roughly
# 20x realtime on this GPU while batched whisper large-v3 is roughly 10x. "fetch" only applies
# when the source is a URL, and the remaining stages are renormalised when it is absent so the
# bar always spans a full 0..1 either way.
STAGE_SHARES = {
    "fetch": 0.10,
    "decode": 0.04,
    "diarize": 0.28,
    "transcribe": 0.62,
    "refine": 0.03,
    "finish": 0.06,
    # A dozen model calls, each writing a section of real prose, plus an optional research pass
    # that waits on web searches. Measured against transcription rather than guessed: on a
    # two-hour podcast at the default rung, decoding takes about twelve minutes and the summary
    # about five, and this share reproduces that ratio.
    "summarise": 0.25,
}


def _pass_cost(decode: dict, model: str) -> float:
    """Roughly how long one voting pass takes, relative to a sequential large-v3 decode.

    Only the ratios matter — they set each pass's share of the progress bar. Calibrated against
    the nine-pass run of 2026-08-07, where the passes took between one and ten minutes each:
    batched and turbo are the cheap ones, a widened beam is far and away the expensive one.
    """
    cost = 1.0
    if "turbo" in model:
        cost *= 0.4                                  # four decoder layers instead of thirty-two
    if str(decode.get("asr_mode")) == "fast":
        cost *= 0.4                                  # batched: measured 2.5x faster
    beam = float(decode.get("beam_size") or 5)
    if beam > 5:
        cost *= beam / 5                             # beam 20 measured ~3.3x a beam-5 decode
    return max(cost, 0.05)


def stage_spans(with_fetch: bool, with_refine: bool = True,
                with_summary: bool = False) -> dict[str, tuple[float, float]]:
    """Map each stage to its (start, width) on the 0..1 progress bar."""
    shares = dict(STAGE_SHARES)
    if not with_fetch:
        shares.pop("fetch")
    if not with_refine:
        shares.pop("refine")
    if not with_summary:
        shares.pop("summarise")
    total = sum(shares.values())
    spans: dict[str, tuple[float, float]] = {}
    acc = 0.0
    for name, share in shares.items():
        width = share / total
        spans[name] = (acc, width)
        acc += width
    return spans

# Escalating fallbacks if a stage dies with CUDA OOM. The MiB figures are the total VRAM the
# stage was measured to occupy on this box (large-v3), not estimates: batch 16 really does reach
# ~9.1 GB, which is why it is gated behind a lot of headroom rather than used by default.
ASR_TIERS = [
    ("cuda", "float16", 16),       # ~9.1 GB measured
    ("cuda", "float16", 8),        # ~5.5 GB
    ("cuda", "float16", 4),        # ~4.0 GB
    ("cuda", "int8_float16", 4),   # ~2.5 GB
    ("cuda", "int8_float16", 1),   # ~1.8 GB
    ("cpu", "int8", 1),            # ~0.5x realtime, last resort
]


class Cancelled(RuntimeError):
    pass


class StageError(RuntimeError):
    def __init__(self, msg: str, kind: str = "runtime"):
        super().__init__(msg)
        self.kind = kind


@dataclass
class JobSpec:
    """A job's source is either a local file or a URL; exactly one is used.

    When `url` is set, `input_path` is ignored and filled in by the fetch stage.
    """
    input_path: Path | None
    output_dir: Path
    cfg: dict = field(default_factory=dict)
    url: str | None = None


def free_vram_mib() -> int:
    """Free VRAM in MiB, or -1 if it cannot be determined (then we just assume GPU is fine)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False)
        if out.returncode == 0:
            return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    return -1


def gpu_tenants() -> list[str]:
    """Names of processes currently holding VRAM — shown in the low-VRAM banner."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False)
        if out.returncode != 0:
            return []
        names = []
        for line in out.stdout.strip().splitlines():
            if not line.strip():
                continue
            name, _, mem = line.partition(",")
            names.append(f"{Path(name.strip()).name} ({mem.strip()} MiB)")
        return names
    except (OSError, subprocess.TimeoutExpired):
        return []


def pick_asr_tier(free_mib: int) -> int:
    """Index into ASR_TIERS based on currently free VRAM.

    Thresholds carry ~1 GB of headroom above each tier's measured footprint, because the cost of
    guessing high is not a graceful degradation — it is discovering the OOM several minutes into
    a multi-hour transcription and starting that stage over.
    """
    if free_mib < 0:
        return 1
    if free_mib >= 10_200:
        return 0
    if free_mib >= 6_500:
        return 1
    if free_mib >= 5_000:
        return 2
    if free_mib >= 3_400:
        return 3
    if free_mib >= 2_400:
        return 4
    return 5


def _asr_env() -> dict:
    """ctranslate2 does NOT auto-discover the pip CUDA wheels: without these on the loader path
    the model constructs fine and then throws 'libcublas.so.12 is not found' on first encode.
    (Same shim as local-wisprflow/wf-run:10.)"""
    env = dict(os.environ)
    sp = PROJECT_ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
    parts = [str(sp / "nvidia" / "cublas" / "lib"), str(sp / "nvidia" / "cudnn" / "lib")]
    if env.get("LD_LIBRARY_PATH"):
        parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
    env["HF_HUB_OFFLINE"] = "1"
    return env


def _diar_env() -> dict:
    """Environment for the torch/cu13 worker.

    LD_LIBRARY_PATH is deliberately STRIPPED. The launcher sets it to the ASR venv's CUDA-12
    wheels, and cuDNN 12 and cuDNN 13 builds share the same SONAME (libcudnn.so.9) — so an
    inherited path would make torch load a cuDNN built against the wrong CUDA major. torch finds
    its own bundled libraries through RPATH and needs no help.
    """
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env["HF_HUB_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_stem(title: str, limit: int = 120) -> str:
    """Turn a video title into a filename stem that is safe on any filesystem.

    Episode titles routinely contain slashes, colons, pipes and emoji; used verbatim they would
    either create stray directories or fail the write outright.
    """
    stem = _UNSAFE_FILENAME.sub("-", (title or "").strip())
    stem = re.sub(r"\s+", " ", stem).strip(" .-")
    if len(stem) > limit:
        stem = stem[:limit].rstrip(" .-")
    return stem or "transcript"


def _job_key(path: Path) -> str:
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


class Runner:
    """Executes one transcription job. Instantiate, call run(), optionally cancel()."""

    def __init__(
        self,
        spec: JobSpec,
        on_progress: Callable[[float, str, str], None] | None = None,
        on_log: Callable[[str], None] | None = None,
    ):
        self.spec = spec
        self.cfg = spec.cfg
        self._on_progress = on_progress or (lambda *_: None)
        self._on_log = on_log or (lambda _: None)
        self._cancel = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._base = 0.0
        self._weight = 0.0
        self._stage_base = 0.0
        self._stage_weight = 0.0
        self._phase = ""
        self._t_phase = 0.0
        self._meta: fetch.MediaMeta | None = None
        # What this run served from disk instead of computing. A resumed job is fast for reasons
        # that have nothing to do with the quality setting, and reporting its wall clock as a
        # realtime factor makes the slider look like it was ignored — 29.5x on a rung that says
        # 10.7x, because three cached passes cost nothing and only the diarization re-ran.
        self._reused: list[str] = []
        # Seconds actually spent decoding, and how many passes that covers. The rungs' measured
        # figures are transcription-only (diarization cached), so this is the number that is
        # comparable with them; end-to-end wall clock is not.
        self._asr_seconds = 0.0
        self._asr_ran = 0
        self._asr_total = 0

    # -- control ---------------------------------------------------------------------
    def cancel(self) -> None:
        self._cancel.set()
        p = self._proc
        if p and p.poll() is None:
            p.kill()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def _check(self) -> None:
        if self._cancel.is_set():
            raise Cancelled("cancelled by user")

    # -- progress plumbing -----------------------------------------------------------
    def log(self, msg: str) -> None:
        self._on_log(msg)

    def _stage(self, phase: str, base: float, weight: float) -> None:
        self._phase, self._base, self._weight = phase, base, weight
        # The stage's own span, kept apart from _base/_weight because `_subspan` narrows those.
        self._stage_base, self._stage_weight = base, weight
        self._t_phase = time.time()
        self._on_progress(base, phase, "")

    @contextmanager
    def _subspan(self, index: int, weights: list[float]) -> Iterator[None]:
        """Confine progress reports to slice `index` of the current stage.

        Voting runs the same decode up to nine times, and each of those reports its own 0..1.
        Without this the bar would fill and snap back to the start once per pass, and the ETA —
        elapsed time divided by the fraction done — would promise the job was nearly over every
        time a pass ended. `weights` gives each pass its share of the stage, because the passes
        are not the same size: a batched decode is a couple of minutes where beam 20 is ten.
        """
        total = sum(weights) or 1.0
        base, weight = self._base, self._weight
        self._base = base + weight * (sum(weights[:index]) / total)
        self._weight = weight * (weights[index] / total)
        try:
            yield
        finally:
            self._base, self._weight = base, weight

    def _frac(self, f: float, detail: str = "") -> None:
        f = max(0.0, min(1.0, f))
        here = self._base + self._weight * f
        if not detail:
            # Estimate from progress through the whole STAGE, not through the current subspan,
            # so a nine-pass decode counts down once rather than nine times.
            done = ((here - self._stage_base) / self._stage_weight) if self._stage_weight else f
            if done > 0.02:
                elapsed = time.time() - self._t_phase
                if elapsed > 5:
                    remain = elapsed * (1 - done) / done
                    detail = f"~{human_duration(remain)} left"
        self._on_progress(here, self._phase, detail)

    # -- subprocess driver -----------------------------------------------------------
    def _run_worker(self, python: Path, module: str, job: dict, env: dict) -> Iterator[dict]:
        """Spawn a worker and yield its JSON events. stderr is drained on a thread so a chatty
        library can never deadlock by filling the pipe buffer."""
        if not python.exists():
            raise StageError(f"missing interpreter {python} — run install.sh", kind="setup")
        cmd = [str(python), "-m", module]
        self._proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT), env=env, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc = self._proc

        def drain() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                line = line.rstrip()
                if line and "warn" not in line.lower():
                    self._on_log(line)

        t = threading.Thread(target=drain, daemon=True)
        t.start()
        # A worker that fails emits an `error` event and *then* exits non-zero. The rc check below
        # runs inside this generator, i.e. before the caller's loop body ever sees the last event,
        # so the diagnosis has to be captured here — otherwise it is replaced by a bare "exited
        # with code 1" and the caller's `kind` dispatch (the ASR OOM tier ladder) never fires.
        last_err: dict | None = None
        try:
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps(job) + "\n")
            proc.stdin.flush()
            proc.stdin.close()
            for line in proc.stdout:
                if self._cancel.is_set():
                    proc.kill()
                    raise Cancelled("cancelled by user")
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    self._on_log(line)  # stray stdout print from a library
                    continue
                if isinstance(ev, dict) and ev.get("ev") == "error":
                    last_err = ev
                yield ev
            rc = proc.wait(timeout=120)
            # cancel() kills the child from another thread, which simply closes stdout and ends
            # the loop above without an exception. Without this check the stage would return its
            # partial results as if they were complete — and go on to persist them as a
            # checkpoint, so a later resume would silently reuse a truncated (often empty) result.
            if self._cancel.is_set():
                raise Cancelled("cancelled by user")
            if rc != 0:
                if last_err:
                    raise StageError(str(last_err.get("msg") or f"{module} exited with code {rc}"),
                                     kind=str(last_err.get("kind") or "runtime"))
                raise StageError(f"{module} exited with code {rc}")
        finally:
            t.join(timeout=2)
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
            self._proc = None

    # -- stages ----------------------------------------------------------------------
    def _decode(self, info: audio.MediaInfo, cache: Path, chain: str | None = None) -> Path:
        # Conditioned audio gets its own filename so the untouched decode stays cached alongside
        # it — a sweep that compares filters would otherwise re-decode the source every time.
        # `chain` is passed explicitly by the voting passes: a pass that asks for filtered audio
        # has to GET filtered audio. Reading the filter off self.cfg here was a silent no-op for
        # them, because only the extra passes ever set one, so the denoise and clean passes
        # decoded the untouched WAV and returned output byte-identical to the pivot — duplicate
        # ballots dressed up as independent opinions.
        if chain is None:
            chain = str(presets.decode_settings(self.cfg).get("audio_filter") or "")
        wav = cache / (f"decoded-{chain}.wav" if chain else "decoded.wav")
        if wav.exists() and wav.stat().st_size > 1024:
            self.log(f"reusing decoded audio ({wav.stat().st_size / 1e6:.0f} MB)")
            self._reused.append(f"decoded audio ({chain})" if chain else "decoded audio")
            self._frac(1.0)
            return wav
        self.log(f"decoding {info.codec} -> 16 kHz mono WAV"
                 + (f" ({chain} filter)" if chain else ""))
        audio.decode_to_wav(info.path, wav, info.duration,
                            on_progress=lambda f: self._frac(f),
                            should_cancel=self.cancelled,
                            filter_chain=chain)
        return wav

    def _diarize(self, wav: Path, cache: Path) -> list[DiarTurn]:
        settings = {
            "pipeline": self.cfg.get("diar_pipeline"),
            "speaker_mode": self.cfg.get("speaker_mode"),
            "num_speakers": self.cfg.get("num_speakers"),
            "min_speakers": self.cfg.get("min_speakers"),
            "max_speakers": self.cfg.get("max_speakers"),
            # Diarization runs on the same WAV the recogniser gets, so conditioning the audio
            # changes its turns too. Without this key a filtered run would silently reuse turns
            # computed from the untouched audio.
            "audio_filter": presets.decode_settings(self.cfg).get("audio_filter") or "",
        }
        ck = cache / "diar.json"
        if ck.exists():
            try:
                data = json.loads(ck.read_text())
                if data.get("settings") == settings:
                    turns = [DiarTurn(**t) for t in data["turns"]]
                    self.log(f"reusing diarization checkpoint ({len(turns)} turns)")
                    self._reused.append("diarization")
                    self._frac(1.0)
                    return turns
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        job = dict(settings)
        job.update({"wav": str(wav), "hf_token": hf_token(), "device": "cuda"})
        turns: list[DiarTurn] = []
        err: dict | None = None
        for ev in self._run_worker(DIAR_PYTHON, "sinribe.workers.diar_worker", job, _diar_env()):
            kind = ev.get("ev")
            if kind == "ready":
                self.log(f"diarization on {ev.get('device')} — {ev.get('pipeline')}")
            elif kind == "progress":
                self._frac(float(ev.get("done", 0.0)), str(ev.get("step") or ""))
            elif kind == "result":
                turns = [DiarTurn(**t) for t in ev.get("turns", [])]
            elif kind == "error":
                err = ev
        if err:
            raise StageError(err.get("msg", "diarization failed"), kind=err.get("kind", "runtime"))
        self._check()  # never persist a checkpoint from an interrupted stage

        ck.write_text(json.dumps({"settings": settings,
                                  "turns": [t.__dict__ for t in turns]}))
        return turns

    def _refine(self, wav: Path, words: list[Word], diar_turns: list[DiarTurn],
                cache: Path) -> list[SentenceScore]:
        """Score every sentence against the speakers' voice prints; [] means 'no opinion'.

        Failures here are deliberately non-fatal: the stage only ever *improves* attribution, so a
        missing venv or an unexpected model error should cost accuracy, not the transcript.
        """
        spans = sentence_spans(words)
        if not spans or len({t.speaker for t in diar_turns}) < 2:
            self._frac(1.0)
            return []

        sentences = [{"start": round(words[s[0]].start, 3), "end": round(words[s[-1]].end, 3)}
                     for s in spans]
        # Keyed on the sentence BOUNDARIES, not just how many there are. Two decodes of the same
        # audio routinely produce the same sentence count with different spans, and a count-only
        # key would then hand the second one voice-print scores computed for the first one's
        # audio — invisible, and exactly the kind of thing a config sweep would blame on the knob
        # it was testing.
        digest = hashlib.sha1(
            json.dumps(sentences, sort_keys=True).encode()).hexdigest()[:16]
        settings = {"pipeline": self.cfg.get("diar_pipeline"), "n": len(sentences),
                    "spans": digest}
        ck = cache / "refine.json"
        if ck.exists():
            try:
                data = json.loads(ck.read_text())
                if data.get("settings") == settings:
                    scores = [SentenceScore(**s) for s in data["scores"]]
                    self.log(f"reusing speaker-refinement checkpoint ({len(scores)} sentences)")
                    self._reused.append("speaker refinement")
                    self._frac(1.0)
                    return scores
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        job = {"wav": str(wav), "sentences": sentences,
               "turns": [t.__dict__ for t in diar_turns],
               "pipeline": self.cfg.get("diar_pipeline"), "hf_token": hf_token(),
               "device": "cuda"}
        scores: list[SentenceScore] = []
        try:
            for ev in self._run_worker(DIAR_PYTHON, "sinribe.workers.refine_worker",
                                       job, _diar_env()):
                kind = ev.get("ev")
                if kind == "progress":
                    self._frac(float(ev.get("done", 0.0)))
                elif kind == "result":
                    scores = [SentenceScore(speaker=s.get("speaker"),
                                            margin=float(s.get("margin") or 0.0))
                              for s in ev.get("scores", [])]
                elif kind == "error":
                    self.log(f"speaker refinement unavailable: {ev.get('msg')}")
        except Cancelled:
            raise
        except StageError as e:
            self.log(f"speaker refinement skipped: {e}")
            return []

        self._check()  # never persist a checkpoint from an interrupted stage
        if scores:
            ck.write_text(json.dumps({"settings": settings,
                                      "scores": [s.__dict__ for s in scores]}))
        return scores

    def _transcribe(self, wav: Path, duration: float, cache: Path,
                    info: audio.MediaInfo | None = None) -> tuple[list[Word], dict]:
        """Decode the audio, voting across several passes when the quality rung asks for it."""
        rung = presets.for_target(self.cfg.get("speed_target", 20))
        extra = list(getattr(rung, "extra_passes", ()) or ())
        if not extra:
            self._asr_total = 1
            return self._transcribe_once(wav, duration, cache)

        # Several decodes, then a word-by-word majority vote. This is the only thing measured
        # that converts more compute into fewer errors: 21.3 % for one pass against 20.1 % for
        # five, because the passes chunk the audio differently and so make *different* mistakes.
        passes: list[list[Word]] = []
        asr_info: dict = {}
        base_overrides = dict(self.cfg.get("decode_overrides") or {})
        base_chain = str(presets.decode_settings(self.cfg).get("audio_filter") or "")
        plans = []
        seen_plans: dict[tuple, int] = {}
        for override in [{}] + extra:
            job_cfg = dict(self.cfg)
            job_cfg["decode_overrides"] = {**base_overrides, **override}
            # Two passes that resolve to the same model and the same decode settings produce the
            # same words, and a duplicate ballot does not break a tie — it doubles one opinion's
            # weight and outvotes the passes that disagree with it. This happens for real: pin a
            # model in the dropdown and the "different model" pass silently becomes a re-run of
            # the pivot. Drop it rather than pay GPU minutes to corrupt the vote.
            key = (presets.model_for(job_cfg),
                   tuple(sorted(presets.decode_settings(job_cfg).items())))
            if key in seen_plans:
                self.log(f"pass {len(plans) + 1} dropped — identical to pass "
                         f"{seen_plans[key] + 1} ({key[0]}); a duplicate would just double that "
                         f"pass's vote")
                continue
            seen_plans[key] = len(plans)
            plans.append(job_cfg)
        # Give the bar each pass's real share up front, so it advances evenly instead of racing
        # through the batched passes and then appearing to hang for ten minutes on beam 20.
        costs = [_pass_cost(presets.decode_settings(c), presets.model_for(c)) for c in plans]
        self._asr_total = len(plans)

        for i, job_cfg in enumerate(plans):
            self._check()
            # A pass may name a model the user never installed. Skipping it costs a little
            # accuracy; failing the whole job over an optional extra decode would be absurd.
            if not model_available(presets.model_for(job_cfg)):
                self.log(f"pass {i + 1} skipped — {presets.model_for(job_cfg)} is not installed "
                         f"(run tools/convert_german_models.sh for the full quality range)")
                self._asr_total -= 1
                continue
            with self._subspan(i, costs):
                # Naming the pass in the phase label is what tells the user that a bar which has
                # been moving for twenty minutes is working, not stuck.
                label = (f" · pass {i + 1}/{len(plans)}" if len(plans) > 1 else "")
                # A pass that asks for conditioned audio gets its own WAV. The filter is part of
                # what that pass costs, so it is timed with it.
                chain = str(presets.decode_settings(job_cfg).get("audio_filter") or "")
                pass_wav = wav
                if chain != base_chain:
                    if info is None:
                        self.log(f"pass {i + 1} skipped — cannot apply the {chain} filter without "
                                 f"the source audio")
                        self._asr_total -= 1
                        continue
                    self._phase = f"Filtering audio{label}"
                    self._frac(0.0)
                    t_filter = time.time()
                    pass_wav = self._decode(info, cache, chain=chain)
                    self._asr_seconds += time.time() - t_filter
                self._phase = f"Transcribing{label}"
                self._frac(0.0)
                words, pass_info = self._transcribe_once(
                    pass_wav, duration, cache, cfg=job_cfg, tag=f"pass{i}")
            self.log(f"pass {i + 1}/{len(plans)}: {len(words)} words")
            passes.append(words)
            if not asr_info:
                asr_info = pass_info

        merged = rover.combine(passes)
        agree = rover.agreement(passes)
        self.log(f"voted {len(passes)} passes -> {len(merged)} words "
                 f"({agree:.0%} of positions were unanimous)")
        asr_info = dict(asr_info)
        asr_info.update({"passes": len(passes), "vote_agreement": round(agree, 4)})
        return merged, asr_info

    def _transcribe_once(self, wav: Path, duration: float, cache: Path,
                         cfg: dict | None = None, tag: str = "") -> tuple[list[Word], dict]:
        t_pass = time.time()
        cfg = self.cfg if cfg is None else cfg
        decode = presets.decode_settings(cfg)
        model_name = presets.model_for(cfg)
        accurate = str(decode["asr_mode"]) == "accurate"
        # The checkpoint key. EVERY setting that can change the words must appear here. A knob
        # that is passed to the worker but left out of this dict makes a re-run silently reuse
        # the previous result — during a config sweep that shows up as every variant scoring
        # identically, which reads like "the setting does nothing" rather than like a bug.
        hotwords = (cfg.get("hotwords") or "").strip()
        if hotwords and not supports_hotwords(model_name):
            # Dropped rather than fatal: the transcript is what the user came for, and this
            # combination would otherwise return an empty one. Said loudly, because silently
            # ignoring names the user typed is its own kind of wrong.
            self.log(f"WARNING: {model_name} returns an empty transcript when given hotwords, "
                     f"so 'Names & terms' is being ignored for this run. Use large-v3 or "
                     f"large-v3-turbo-german if you need it.")
            hotwords = ""

        settings = {
            "model": model_name,
            "language": cfg.get("language"),
            "vad_filter": cfg.get("vad_filter"),
            "hotwords": hotwords,
            **{k: decode[k] for k in sorted(decode)},
        }
        ck = cache / (f"asr-{tag}.json" if tag else "asr.json")
        if ck.exists():
            try:
                data = json.loads(ck.read_text())
                if data.get("settings") == settings:
                    words = [Word(**w) for w in data["words"]]
                    self.log(f"reusing transcription checkpoint ({len(words)} words)")
                    self._reused.append(f"transcription {tag}" if tag else "transcription")
                    self._frac(1.0)
                    return words, data.get("info", {})
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        lang = cfg.get("language")
        tier = pick_asr_tier(free_vram_mib())
        last_err: StageError | None = None

        while tier < len(ASR_TIERS):
            device, compute_type, batch = ASR_TIERS[tier]
            if model_name.startswith("large-v3") and device == "cpu":
                self.log("WARNING: falling back to CPU — large-v3 runs ~0.5x realtime there")
            job = {
                "wav": str(wav), "duration": duration,
                "model": resolve_model(model_name),
                "device": device, "compute_type": compute_type,
                # Accurate mode decodes sequentially. Batching splits the audio at VAD boundaries
                # and decodes the pieces independently, which is where words at chunk edges go
                # missing and where word timestamps lose the precision attribution needs.
                "batch_size": 1 if accurate else batch,
                "sequential": accurate,
                "language": None if lang in (None, "", "auto") else lang,
                "vad_filter": bool(cfg.get("vad_filter", True)),
                "hotwords": settings["hotwords"],
                "beam_size": int(decode["beam_size"]),
                "patience": float(decode["patience"]),
                "condition_on_previous_text": bool(decode["condition_on_previous_text"]),
                "repetition_penalty": float(decode["repetition_penalty"]),
                "no_repeat_ngram_size": int(decode["no_repeat_ngram_size"]),
                "prompt_reset_on_temperature": float(decode["prompt_reset_on_temperature"]),
                "temperature_fallback": bool(decode["temperature_fallback"]),
                "vad_min_silence_ms": int(decode["vad_min_silence_ms"]),
                "vad_speech_pad_ms": int(decode["vad_speech_pad_ms"]),
            }
            words: list[Word] = []
            info: dict = {"device": device, "compute_type": compute_type,
                          "batch_size": job["batch_size"], "asr_mode": decode["asr_mode"],
                          "model": model_name, "decode": dict(decode)}
            err: dict | None = None
            try:
                for ev in self._run_worker(ASR_PYTHON, "sinribe.workers.asr_worker",
                                           job, _asr_env()):
                    kind = ev.get("ev")
                    if kind == "ready":
                        self.log(f"whisper {ev.get('model')} on {ev.get('device')} "
                                 f"({ev.get('compute_type')}, batch {ev.get('batch_size')})")
                    elif kind == "info":
                        info.update({k: ev.get(k) for k in
                                     ("language", "language_probability", "duration")})
                        self.log(f"language: {ev.get('language')} "
                                 f"({float(ev.get('language_probability') or 0):.2f})")
                    elif kind == "segment":
                        for w in ev.get("words", []):
                            words.append(Word(start=w["start"], end=w["end"],
                                              word=w["word"], prob=w.get("prob", 0.0)))
                    elif kind == "progress":
                        total = float(ev.get("total") or duration or 1.0)
                        self._frac(float(ev.get("done", 0.0)) / total if total else 0.0)
                    elif kind == "error":
                        err = ev
                if err:
                    raise StageError(err.get("msg", "transcription failed"),
                                     kind=err.get("kind", "runtime"))
            except StageError as e:
                if e.kind == "oom" and tier + 1 < len(ASR_TIERS):
                    self.log(f"CUDA out of memory at tier {tier} — retrying smaller")
                    tier += 1
                    last_err = e
                    continue
                raise
            self._check()  # never persist a checkpoint from an interrupted stage
            ck.write_text(json.dumps({"settings": settings, "info": info,
                                      "words": [w.__dict__ for w in words]}))
            # Only decodes that actually ran count toward the speed this job is reported at.
            self._asr_seconds += time.time() - t_pass
            self._asr_ran += 1
            return words, info

        raise last_err or StageError("transcription failed")

    def _fetch(self) -> Path:
        """Download the URL's audio into the shared download cache."""
        url = str(self.spec.url)
        cookies = str(self.cfg.get("yt_cookies_from_browser") or "") or None
        try:
            if self._meta is None:
                self.log(f"looking up {url}")
                self._meta = fetch.probe(url, cookies)
            meta = self._meta
            self.log(f"{meta.title}"
                     + (f" — {meta.uploader}" if meta.uploader else "")
                     + (f" ({human_duration(meta.duration)})" if meta.duration else ""))
            path, meta = fetch.download_audio(
                url, DOWNLOADS_DIR, meta=meta, cookies_from_browser=cookies,
                on_progress=lambda f, d: self._frac(f, d),
                should_cancel=self.cancelled)
        except fetch.FetchCancelled as e:
            raise Cancelled("cancelled by user") from e
        except fetch.FetchError as e:
            raise StageError(str(e), kind="fetch") from e
        self._meta = meta
        self.log(f"downloaded {path.name} ({path.stat().st_size / 1e6:.0f} MB)")
        return path

    # -- main ------------------------------------------------------------------------
    def run(self) -> dict:
        t_start = time.time()
        out_dir = Path(self.spec.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        refine = bool(self.cfg.get("refine_speakers", True))
        want_summary = bool(self.cfg.get("web_summary"))
        spans = stage_spans(with_fetch=bool(self.spec.url), with_refine=refine,
                            with_summary=want_summary)

        # Probed before a single second of audio is decoded. The alternative -- discovering a
        # missing API key in the stage AFTER the transcription -- means the user waits out the
        # whole job to be told the thing they ticked the box for cannot run. The probe is free:
        # it imports the SDK and constructs a client, it does not call the API.
        summary_provider = None
        if want_summary:
            try:
                summary_provider = summary_mod.pick(self.cfg)
            except summary_mod.Unavailable as e:
                raise StageError(f"the summary cannot be generated: {e}", kind="setup") from e
            skipped = list(getattr(summary_provider, "skipped", []) or [])
            for reason in skipped:
                self.log(f"WARNING: summary provider not available, skipped: {reason}")
            if self.cfg.get("summary_research") and not summary_provider.supports_research:
                # Refused at second zero rather than quietly writing a brief nobody checked.
                # The user asked for the facts to be checked against the web; a page that
                # silently skipped that looks exactly like one that did it and found nothing.
                raise StageError(
                    f"web research was requested, but {summary_provider.label} cannot search "
                    f"the web" + (f" ({'; '.join(skipped)})" if skipped else "")
                    + ". Fix the Claude provider, or untick 'Verify with web research'.",
                    kind="setup")
            self.log(f"summary will be written by {summary_provider.label}"
                     + (" with web research" if self.cfg.get("summary_research") else ""))

        if self.spec.url:
            self._stage("Downloading audio", *spans["fetch"])
            src = self._fetch()
        else:
            if not self.spec.input_path:
                raise StageError("no input file or URL given")
            src = Path(self.spec.input_path)

        model_name = presets.model_for(self.cfg)
        if not model_available(model_name):
            raise StageError(
                f"the {model_name} model is not installed — run tools/convert_german_models.sh, "
                f"or pick a different model. (Workers run offline, so it cannot be fetched here.)",
                kind="setup")

        self._stage("Analysing", *spans["decode"])
        info = audio.probe(src)
        self.log(f"{src.name}: {info.codec}, {info.channels}ch @ {info.sample_rate} Hz, "
                 f"{human_duration(info.duration)}")
        if not info.has_audio:
            raise StageError("no usable audio stream in this file")

        cache = JOBS_DIR / _job_key(src)
        cache.mkdir(parents=True, exist_ok=True)

        self._check()
        self._stage("Decoding audio", *spans["decode"])
        wav = self._decode(info, cache)

        self._check()
        self._stage("Identifying speakers", *spans["diarize"])
        diar_turns = self._diarize(wav, cache)
        n_spk = len({t.speaker for t in diar_turns})
        self.log(f"diarization: {len(diar_turns)} turns, {n_spk} speaker(s)")

        self._check()
        self._stage("Transcribing", *spans["transcribe"])
        words, asr_info = self._transcribe(wav, info.duration, cache, info)
        self.log(f"transcription: {len(words)} words")

        # A collapsed decode still exits 0 and still writes a tidy transcript, so nothing else in
        # the pipeline notices. Measured once: large-v3-german with context conditioning returned
        # 156 words for a 65-minute interview and reported success.
        coverage = speech_coverage(words, diar_turns)
        if coverage < 0.5:
            self.log(f"WARNING: only {coverage:.0%} of the detected speech became words — "
                     f"the transcript is probably truncated. Try a different quality setting "
                     f"or model.")

        scores: list[SentenceScore] = []
        if refine:
            self._check()
            self._stage("Checking speakers", *spans["refine"])
            scores = self._refine(wav, words, diar_turns, cache)
            if scores:
                decisive = sum(1 for s in scores
                               if s.speaker and s.margin >= float(
                                   self.cfg.get("refine_margin", 0.15)))
                self.log(f"voice-print check: {decisive}/{len(scores)} sentences decisive")

        self._check()
        base, fin = spans["finish"]
        self._stage("Aligning speakers", base, fin * 0.2)
        turns, stats = merge(
            words, diar_turns,
            turn_gap_s=float(self.cfg.get("turn_gap_s", 1.5)),
            max_turn_chars=int(self.cfg.get("max_turn_chars", 1200)),
            flicker_min_words=int(self.cfg.get("flicker_min_words", 3)),
            orphan_window_s=float(self.cfg.get("orphan_word_window_s", 2.0)),
            sentence_atomic=bool(self.cfg.get("sentence_atomic", True)),
            sentence_scores=scores or None,
            refine_margin=float(self.cfg.get("refine_margin", 0.15)),
            refine_min_seconds=float(self.cfg.get("refine_min_seconds", 0.8)),
        )
        self._frac(1.0)
        self.log(f"merged into {len(turns)} turns across {len(stats)} speaker(s)")

        enrichment: dict = {}
        if self.cfg.get("llm_enrich"):
            self._check()
            self._stage("Summarising", base + fin * 0.2, fin * 0.6)
            self.log(f"asking {self.cfg.get('llm_model')} for chapters and a summary")
            enrichment = enrich_mod.enrich(
                turns, info.duration,
                url=str(self.cfg.get("llm_url")), model=str(self.cfg.get("llm_model")),
                timeout=float(self.cfg.get("llm_timeout", 180)),
                on_progress=self._frac, should_cancel=self.cancelled,
                language=asr_info.get("language"))
            if not enrichment:
                self.log("LLM enrichment unavailable or empty — writing transcript without it")

        self._check()
        self._stage("Writing files", base + fin * 0.8, fin * 0.2)
        elapsed = time.time() - t_start
        meta = self._meta
        result = {
            # For a downloaded episode the useful "source" is the page it came from, not the
            # opaque <video-id>.webm sitting in the cache.
            "source": meta.webpage_url if meta else str(src),
            "title": meta.title if meta else src.stem.replace("_", " ").replace("-", " ").strip(),
            "uploader": meta.uploader if meta else "",
            "published": meta.published if meta else "",
            "description": meta.description if meta else "",
            "chapters": list(meta.chapters) if meta else [],
            "local_file": str(src),
            "duration": info.duration,
            "language": asr_info.get("language"),
            "language_probability": asr_info.get("language_probability"),
            "model": asr_info.get("model") or self.cfg.get("asr_model"),
            "decode": asr_info.get("decode") or {},
            "speed_target": self.cfg.get("speed_target"),
            "speech_coverage": round(coverage, 4),
            # How many decodes were voted on, and how much they agreed. Without these two a
            # nine-pass transcript is indistinguishable from a one-pass one after the fact,
            # which makes "was the slow setting worth it?" unanswerable from the output alone.
            "passes": asr_info.get("passes", 1),
            "vote_agreement": asr_info.get("vote_agreement"),
            "device": asr_info.get("device"),
            "compute_type": asr_info.get("compute_type"),
            "batch_size": asr_info.get("batch_size"),
            "diar_pipeline": self.cfg.get("diar_pipeline"),
            "elapsed": elapsed,
            "realtime_factor": (info.duration / elapsed) if elapsed else 0.0,
            # End-to-end wall clock answers "how long did I wait", not "how fast is this quality
            # setting". Those differ by whatever was served from the checkpoint cache, and on a
            # re-run they differ by a factor of three. The rungs in presets.py are calibrated on
            # transcription alone, so that is what gets compared with them and folded back into
            # the slider's estimate — and only when every pass this rung asks for actually ran.
            "asr_seconds": round(self._asr_seconds, 2),
            "asr_realtime_factor": ((info.duration / self._asr_seconds)
                                    if self._asr_seconds else 0.0),
            "asr_passes_ran": self._asr_ran,
            "asr_passes_total": self._asr_total,
            "reused_stages": list(self._reused),
            # True when the timing describes this run's own work end to end.
            "timing_is_cold": not self._reused,
            "finished_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

        stem = safe_stem(meta.title) if meta else src.stem
        md_path = out_dir / f"{stem}.md"
        markdown.write(md_path, markdown.render(turns, stats, result,
                                                title=result["title"], enrichment=enrichment,
                                                review=bool(self.cfg.get("review_section", True)),
                                                review_max=int(
                                                    self.cfg.get("review_max_spans", 0))))
        written = [md_path]

        payload = sidecar.build(turns, stats, result, diar_turns=diar_turns,
                                enrichment=enrichment, settings=dict(self.cfg))
        json_path = out_dir / f"{stem}.sinribe.json"
        if self.cfg.get("write_json", True):
            sidecar.write(json_path, payload)
            written.append(json_path)
        if self.cfg.get("write_srt", True):
            written.append(subtitles.write(out_dir / f"{stem}.srt",
                                           subtitles.render_srt(turns)))
        if self.cfg.get("write_vtt", True):
            written.append(subtitles.write(out_dir / f"{stem}.vtt",
                                           subtitles.render_vtt(turns)))

        # Deliberately after the transcript is on disk. Everything above this point is the job
        # the user actually asked for and is now safe; the summary is a second, network-dependent
        # job on top of it, and its failure must cost the page and nothing else.
        if summary_provider is not None:
            self._check()
            self._stage("Writing the summary", *spans["summarise"])
            # Timed separately and reported separately. `elapsed` above means "how long the
            # transcription took" and the transcript header states it; folding several minutes of
            # API calls into that number would make the one figure the user can check disagree
            # with the file it is printed in.
            t_summary = time.time()
            try:
                data = summary_mod.summarize(
                    turns, stats, result, summary_provider,
                    research=bool(self.cfg.get("summary_research")),
                    max_sections=int(self.cfg.get("summary_max_sections", 12)),
                    max_claims=int(self.cfg.get("summary_max_claims", 16)),
                    on_progress=self._frac, on_log=self.log,
                    should_cancel=self.cancelled)
            except summary_mod.Cancelled as e:
                raise Cancelled("cancelled by user") from e
            except Exception as e:  # noqa: BLE001 - the transcript is written; report and go on
                self.log(f"ERROR: the summary could not be written: "
                         f"{type(e).__name__}: {e}")
                result["summary_error"] = f"{type(e).__name__}: {e}"
                result["summary_seconds"] = round(time.time() - t_summary, 2)
            else:
                html_path = webpage.write(out_dir / f"{stem} - Summary.html",
                                          webpage.render_html(data))
                summary_md = webpage.write(out_dir / f"{stem} - Summary.md",
                                           webpage.render_markdown(data))
                written += [html_path, summary_md]
                result["summary"] = data
                if data["meta"].get("research_error"):
                    result["summary_research_error"] = data["meta"]["research_error"]
                result["summary_path"] = str(html_path)
                result["summary_markdown_path"] = str(summary_md)
                result["summary_seconds"] = round(time.time() - t_summary, 2)
                self.log(f"summary: {len(data['sections'])} sections, "
                         f"{len(data.get('sources', []))} source groups in "
                         f"{human_duration(result['summary_seconds'])} -> {html_path.name}")

        if not self.cfg.get("keep_decoded_wav", False):
            # Every decode, not just the base one: a voting rung with conditioned passes leaves a
            # filtered WAV per filter, and at ~120 MB an hour those add up quietly in the job
            # cache long before `cache_days` gets round to them.
            for stale in cache.glob("decoded*.wav"):
                stale.unlink(missing_ok=True)

        self._on_progress(1.0, "Done", "")
        self.log(f"finished in {human_duration(elapsed)}"
                 f"{presets.speed_note(result)} -> {md_path}")

        result.update({
            "turns": turns, "stats": stats, "enrichment": enrichment,
            "markdown_path": str(md_path), "sidecar_path": str(json_path),
            "written": [str(p) for p in written], "sidecar": payload,
        })
        return result


def _span_seconds(spans: list[tuple[float, float]]) -> float:
    """Total time covered by a set of possibly-overlapping intervals."""
    total = 0.0
    end = float("-inf")
    for lo, hi in sorted(spans):
        if hi <= end:
            continue
        total += hi - max(lo, end)
        end = hi
    return total


def speech_coverage(words: list[Word], diar_turns: list[DiarTurn]) -> float:
    """What fraction of the diarized speech ended up as words. 1.0 when there is nothing to say.

    Diarization and transcription are independent views of the same audio, which makes this a
    genuine cross-check rather than a self-report. A decode that collapses — and whisper does
    collapse, silently and with a zero exit code — leaves the diarizer still insisting there are
    sixty-five minutes of speech while the recogniser hands back twenty-six seconds of words.

    Measured on the benchmark interview: healthy decodes score 1.00, the collapsed one 0.008. The
    gap is enormous, so callers should treat this as a catastrophe detector and not read anything
    into the difference between, say, 0.95 and 0.99.
    """
    speech = _span_seconds([(t.start, t.end) for t in diar_turns])
    if speech <= 0:
        return 1.0
    return min(1.0, _span_seconds([(w.start, w.end) for w in words]) / speech)


def purge_old_jobs(days: int = 14) -> int:
    """Delete cached job dirs older than `days`. Returns how many were removed."""
    import shutil
    if not JOBS_DIR.exists():
        return 0
    cutoff = time.time() - days * 86400
    n = 0
    for d in JOBS_DIR.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                n += 1
        except OSError:
            continue
    return n
