# HANDOFF — read this first

Orientation for a future session that does **not** share this one's chat history. The README is
the user-facing doc; this file is the "what is actually proven, what would bite you" companion.

Built 2026-07-26 in one session. Target box: Acer Predator Helios Neo 16, **RTX 5070 Ti Laptop
(12 GB, sm_120 / Blackwell)**, Ubuntu 26.04, driver 595.71.05 / CUDA 13.2, no system CUDA toolkit.

## State: proven vs unproven

| | Status |
|---|---|
| Full pipeline on a local file (decode → diarize → transcribe → merge → render) | ✅ |
| **4 h 00 m file: 10 m 58 s (21.9× realtime)** | ✅ measured, not estimated |
| Speaker attribution vs ground truth | ✅ 9/9 spans, identity stable across the full 4 h |
| Runs with no network (all proxies + `HF_ENDPOINT` blackholed) | ✅ local files |
| Checkpoint resume after cancel | ✅ 1.3 s, byte-identical output |
| Cancel leaves no orphan workers / no poisoned checkpoint | ✅ |
| GUI end-to-end incl. speaker rename + re-export | ✅ |
| YouTube link → download → transcript | ✅ (CC-BY talk; also a wordless film as the empty case) |
| LLM chapters + summary via local Ollama | ✅ `gemma3:4b` on `:11435` |
| Tests / lint | ✅ 88 pytest, ruff clean |
| **A real multi-hour podcast *download*** | ❌ longest real fetch was ~10 min |
| Non-English audio end-to-end | ❌ the German path is only exercised in unit tests |
| `pyannote/speaker-diarization-3.1` pipeline | ❌ licence never accepted — selecting it 403s |

## The one constraint everything else follows from

**`ctranslate2` needs CUDA 12; PyPI `torch 2.13` is a CUDA 13 build. They cannot share a venv.**
Hence `.venv` (ASR + GUI, cu12) and `.venv-diar` (pyannote + torch, cu13), and two worker
*subprocesses* run sequentially.

Two rules that are load-bearing, both learned by failing:

1. `sinribe-run` **must** export `LD_LIBRARY_PATH` with `.venv`'s `nvidia/cublas/lib` and
   `nvidia/cudnn/lib`. Without it the whisper model constructs fine and then throws
   `Library libcublas.so.12 is not found` on the first encode.
2. The diarization worker **must** run with `LD_LIBRARY_PATH` *stripped*
   (`pipeline.py:_diar_env`). cuDNN 12 and cuDNN 13 wheels share the SONAME `libcudnn.so.9`, so
   an inherited cu12 path makes torch load a cuDNN built for the wrong CUDA major.

Splitting also caps peak VRAM at `max(diarize, transcribe)` instead of the sum, because the CUDA
context is only returned on **process exit** — never by in-process unloading. That is what makes
this fit on a 12 GB card that also hosts Ollama (~2.9 GB) and the wisprflow daemon (~2 GB).

## Run it

```bash
sinribe                                             # GUI
sinribe recording.m4a -o ~/Transcripts              # local file, headless
sinribe 'https://youtube.com/watch?v=…' -o ~/Docs   # download, then transcribe
./.venv/bin/python -m pytest tests/ -q              # 88 tests, ~0.05 s, no GPU needed
./.venv/bin/python -m ruff check sinribe/ tests/ --select F,E,W,B --line-length 100
./install.sh --skip-venvs                           # re-install launcher/.desktop only
```

Anything invoking the ASR venv directly (rather than via `sinribe-run`) needs the loader shim:

```bash
SP="$PWD/.venv/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$SP/nvidia/cublas/lib:$SP/nvidia/cudnn/lib"
```

## STOP-don't-fix rules

- **Never** modify or downgrade the NVIDIA driver, kernel, CUDA, or boot config on this machine.
  Everything here is pip-wheel-only and no-sudo by design.
- **Do not** port the adaptive GPU promote/demote monitor from `local-wisprflow`
  (`wf_daemon.py:422-461`). Nexus removed it after its two disagreeing GPU probes thrashed
  placement every ~10 s and leaked host memory until the kernel OOM-killed the daemon
  mid-dictation (journal-proven 2026-07-08). Sinribe loads once per worker life.
- **Do not** pin `yt-dlp`. Sites change delivery constantly; a pinned copy silently rots. If a
  link stops working, `uv pip install --python .venv/bin/python -U yt-dlp` *first*, then debug.
- `pyannote`'s batch sizes look like they default to 1 in the constructor signature — they don't;
  `community-1`'s config sets both to 32. Don't "fix" that.

## Layout

```
sinribe-run              launcher (sets the cu12 loader path); ~/.local/bin/sinribe -> here
install.sh               builds both venvs, fetches models, installs .desktop  (--skip-venvs)
sinribe/
  pipeline.py            orchestrator: stages, progress weights, checkpoints, cancel, VRAM tiers
  merge.py               word -> speaker alignment  ← accuracy lives here
  fetch.py               yt-dlp download + probe    ← the ONLY networked module
  audio.py               ffmpeg probe/decode/clip
  enrich.py              optional Ollama chapters + summary
  textfmt.py             EN/DE sentence splitting, timestamp formatting
  workers/asr_worker.py  faster-whisper subprocess   (.venv,      cu12)
  workers/diar_worker.py pyannote subprocess         (.venv-diar, cu13)
  render/                markdown, subtitles, sidecar
  ui/                    PySide6 window, QThread wrappers, speaker panel
```

State on disk: config `~/.config/sinribe/config.json`; HF token `~/.config/sinribe/hf_token`
(chmod 600, never in the repo); stage checkpoints `~/.cache/sinribe/jobs/<hash>/`; downloaded
media `~/.cache/sinribe/downloads/` (kept indefinitely by default — first place to look when
disk gets tight).

## Design notes worth knowing before changing anything

- **Attribution is word-level, not segment-level.** Whisper segments straddle speaker changes, so
  each *word* goes to the diarization turn it overlaps most; runs shorter than 3 words flanked by
  the same speaker are absorbed (otherwise transcripts ping-pong on every "mhm"). `merge.py`.
- pyannote 4 returns a `DiarizeOutput`, not an `Annotation`. Use
  `.exclusive_speaker_diarization` — no overlapping turns, built for exactly this case.
- The diarization worker enables TF32 (pyannote disables it on import, then prints a hint telling
  you to turn it back on). Matters on multi-hour files.
- VRAM tier thresholds in `pipeline.py` carry ~1 GB of headroom above each tier's *measured*
  footprint (`float16`/batch 16 really does reach ~9.1 GB). Being optimistic here doesn't degrade
  gracefully — it OOMs eight minutes into a four-hour job and redoes the stage.
- A cancelled stage must **never** write a checkpoint. An early bug did, and resume then silently
  produced a transcript with zero speaker separation. Guarded in `_run_worker` and both stages.
- Don't run a URL through `Path()`. It collapses `https://` to `https:/`; that bug shipped a
  broken Source link before it was caught.
- `tests/fixtures/three_speakers.wav` is synthetic (two Piper voices + a JFK archive clip) with
  ground-truth spans in the sibling `.truth.json`, so attribution is checked against a known
  answer rather than by eye. Regenerate: see the fixture script referenced in the PR.

## Suggested first moves

1. Run a **real** 2–4 h podcast link end-to-end — the download half at that scale is the only
   untested part. Transcription at 4 h is already proven.
2. Run a German lecture; confirm the EN/DE sentence splitter and language auto-detect behave.
3. If diarization quality disappoints on real recordings, the knobs are `speaker_mode`
   (auto/exact/range) and `flicker_min_words` / `turn_gap_s` in the config — try pinning the
   speaker count before touching the pipeline.
