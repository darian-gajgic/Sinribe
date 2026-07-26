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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from . import audio, enrich as enrich_mod, fetch
from .config import DOWNLOADS_DIR, JOBS_DIR, hf_token
from .merge import DiarTurn, Word, merge
from .render import markdown, sidecar, subtitles
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
    "finish": 0.06,
}


def stage_spans(with_fetch: bool) -> dict[str, tuple[float, float]]:
    """Map each stage to its (start, width) on the 0..1 progress bar."""
    shares = dict(STAGE_SHARES)
    if not with_fetch:
        shares.pop("fetch")
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
    (Same shim as /home/sinep/local-wisprflow/wf-run:10.)"""
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
        self._phase = ""
        self._t_phase = 0.0
        self._meta: fetch.MediaMeta | None = None

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
        self._t_phase = time.time()
        self._on_progress(base, phase, "")

    def _frac(self, f: float, detail: str = "") -> None:
        f = max(0.0, min(1.0, f))
        if not detail and f > 0.02:
            elapsed = time.time() - self._t_phase
            if elapsed > 5:
                remain = elapsed * (1 - f) / f
                detail = f"~{human_duration(remain)} left"
        self._on_progress(self._base + self._weight * f, self._phase, detail)

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
                    yield json.loads(line)
                except json.JSONDecodeError:
                    self._on_log(line)  # stray stdout print from a library
            rc = proc.wait(timeout=120)
            # cancel() kills the child from another thread, which simply closes stdout and ends
            # the loop above without an exception. Without this check the stage would return its
            # partial results as if they were complete — and go on to persist them as a
            # checkpoint, so a later resume would silently reuse a truncated (often empty) result.
            if self._cancel.is_set():
                raise Cancelled("cancelled by user")
            if rc != 0:
                raise StageError(f"{module} exited with code {rc}")
        finally:
            t.join(timeout=2)
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
            self._proc = None

    # -- stages ----------------------------------------------------------------------
    def _decode(self, info: audio.MediaInfo, cache: Path) -> Path:
        wav = cache / "decoded.wav"
        if wav.exists() and wav.stat().st_size > 1024:
            self.log(f"reusing decoded audio ({wav.stat().st_size / 1e6:.0f} MB)")
            self._frac(1.0)
            return wav
        self.log(f"decoding {info.codec} -> 16 kHz mono WAV")
        audio.decode_to_wav(info.path, wav, info.duration,
                            on_progress=lambda f: self._frac(f),
                            should_cancel=self.cancelled)
        return wav

    def _diarize(self, wav: Path, cache: Path) -> list[DiarTurn]:
        settings = {
            "pipeline": self.cfg.get("diar_pipeline"),
            "speaker_mode": self.cfg.get("speaker_mode"),
            "num_speakers": self.cfg.get("num_speakers"),
            "min_speakers": self.cfg.get("min_speakers"),
            "max_speakers": self.cfg.get("max_speakers"),
        }
        ck = cache / "diar.json"
        if ck.exists():
            try:
                data = json.loads(ck.read_text())
                if data.get("settings") == settings:
                    turns = [DiarTurn(**t) for t in data["turns"]]
                    self.log(f"reusing diarization checkpoint ({len(turns)} turns)")
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

    def _transcribe(self, wav: Path, duration: float, cache: Path) -> tuple[list[Word], dict]:
        settings = {
            "model": self.cfg.get("asr_model"),
            "language": self.cfg.get("language"),
            "beam_size": self.cfg.get("beam_size"),
            "vad_filter": self.cfg.get("vad_filter"),
        }
        ck = cache / "asr.json"
        if ck.exists():
            try:
                data = json.loads(ck.read_text())
                if data.get("settings") == settings:
                    words = [Word(**w) for w in data["words"]]
                    self.log(f"reusing transcription checkpoint ({len(words)} words)")
                    self._frac(1.0)
                    return words, data.get("info", {})
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        lang = self.cfg.get("language")
        tier = pick_asr_tier(free_vram_mib())
        last_err: StageError | None = None

        while tier < len(ASR_TIERS):
            device, compute_type, batch = ASR_TIERS[tier]
            if self.cfg.get("asr_model") == "large-v3" and device == "cpu":
                self.log("WARNING: falling back to CPU — large-v3 runs ~0.5x realtime there")
            job = {
                "wav": str(wav), "duration": duration,
                "model": self.cfg.get("asr_model", "large-v3"),
                "device": device, "compute_type": compute_type, "batch_size": batch,
                "beam_size": int(self.cfg.get("beam_size", 5)),
                "language": None if lang in (None, "", "auto") else lang,
                "vad_filter": bool(self.cfg.get("vad_filter", True)),
            }
            words: list[Word] = []
            info: dict = {"device": device, "compute_type": compute_type, "batch_size": batch}
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
        spans = stage_spans(with_fetch=bool(self.spec.url))

        if self.spec.url:
            self._stage("Downloading audio", *spans["fetch"])
            src = self._fetch()
        else:
            if not self.spec.input_path:
                raise StageError("no input file or URL given")
            src = Path(self.spec.input_path)

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
        words, asr_info = self._transcribe(wav, info.duration, cache)
        self.log(f"transcription: {len(words)} words")

        self._check()
        base, fin = spans["finish"]
        self._stage("Aligning speakers", base, fin * 0.2)
        turns, stats = merge(
            words, diar_turns,
            turn_gap_s=float(self.cfg.get("turn_gap_s", 1.5)),
            max_turn_chars=int(self.cfg.get("max_turn_chars", 1200)),
            flicker_min_words=int(self.cfg.get("flicker_min_words", 3)),
            orphan_window_s=float(self.cfg.get("orphan_word_window_s", 2.0)),
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
                on_progress=self._frac, should_cancel=self.cancelled)
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
            "local_file": str(src),
            "duration": info.duration,
            "language": asr_info.get("language"),
            "language_probability": asr_info.get("language_probability"),
            "model": self.cfg.get("asr_model"),
            "device": asr_info.get("device"),
            "compute_type": asr_info.get("compute_type"),
            "batch_size": asr_info.get("batch_size"),
            "diar_pipeline": self.cfg.get("diar_pipeline"),
            "elapsed": elapsed,
            "realtime_factor": (info.duration / elapsed) if elapsed else 0.0,
            "finished_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

        stem = safe_stem(meta.title) if meta else src.stem
        md_path = out_dir / f"{stem}.md"
        markdown.write(md_path, markdown.render(turns, stats, result,
                                                title=result["title"], enrichment=enrichment))
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

        if not self.cfg.get("keep_decoded_wav", False):
            wav.unlink(missing_ok=True)

        self._on_progress(1.0, "Done", "")
        rtf = result["realtime_factor"]
        self.log(f"finished in {human_duration(elapsed)} ({rtf:.1f}x realtime) -> {md_path}")

        result.update({
            "turns": turns, "stats": stats, "enrichment": enrichment,
            "markdown_path": str(md_path), "sidecar_path": str(json_path),
            "written": [str(p) for p in written], "sidecar": payload,
        })
        return result


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
