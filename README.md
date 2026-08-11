# Sinribe

Transcription with speaker labels. Point it at a lecture or interview recording — or paste a
YouTube podcast link — and get back a readable Markdown transcript where every sentence is
attributed to *Person 1*, *Person 2*, *Person 3*… One speaker or five, ten minutes or four hours.

All the *processing* runs on this machine: no audio is ever uploaded, and no transcript leaves
the box. The single exception is downloading a link you paste, which obviously needs the network;
everything after the download is local.

```
sinribe                                            # GUI
sinribe recording.m4a -o ~/Transcripts             # a local file, headless
sinribe 'https://youtube.com/watch?v=…' -o ~/Docs  # download a podcast, then transcribe it
```

## What it produces

For `vorlesung.m4a` you get `vorlesung.md`, plus optionally `vorlesung.sinribe.json`,
`vorlesung.srt` and `vorlesung.vtt`:

```markdown
# vorlesung

**Duration** 2:14:33 · **Speakers** 3 · **Language** de (0.98)
**Model** large-v3 · CUDA · float16 · **Diarization** speaker-diarization-community-1
**Source** `/home/sinep/Recordings/vorlesung.m4a`
**Transcribed** 2026-07-26 18:03 · 14m 21s (9.4× realtime)

| Speaker | Talk time | Share |
|---|---|---|
| Person 1 | 1:18:02 | 58% |
| Person 2 | 0:44:10 | 33% |
| Person 3 | 0:12:21 | 9% |

---

**[00:00:04] Person 1**
Willkommen zur heutigen Vorlesung.
Heute sprechen wir über Systemtheorie.

**[00:00:19] Person 2**
Kurze Frage vorab — welches Skript gilt?
```

One sentence per line, so diffs and quotes stay clean. Turns break on speaker change, on silences
longer than 1.5 s, and before any single turn grows past ~1200 characters — otherwise a lecturer
holding the floor for forty minutes becomes one unreadable wall of text.

## Pasting a link

Paste into the URL box and hit **Check** to see the title, channel and length before committing
to a download — useful when a "short clip" turns out to be three hours. Or just hit Start; it
looks the link up itself.

- Handled by **yt-dlp**, so YouTube is only the common case — anything yt-dlp supports works.
- The best audio-only stream is kept **in its native container, never re-encoded**. The decode
  stage normalises it anyway, so re-encoding would only add a lossy generation.
- A `watch?v=X&list=Y` link grabs **just that episode**, not the whole playlist.
- Downloads are cached in `~/.cache/sinribe/downloads/` keyed by the site's video id, so
  re-running a link — or resuming a cancelled job — never downloads twice.
- The transcript is named after the video title (sanitised for the filesystem), and its
  **Source** line links back to the page.
- Live streams are rejected with a clear message; wait until they have finished.

For age-restricted or members-only episodes, set the browser to lift cookies from:

```jsonc
// ~/.config/sinribe/config.json
"yt_cookies_from_browser": "firefox"   // or "chrome", "chromium", "edge", "brave"
```

`yt-dlp` is deliberately **unpinned** in `requirements-asr.txt`. Sites change their delivery
constantly and a pinned copy silently rots — if a link stops working, update it first:

```bash
uv pip install --python .venv/bin/python -U yt-dlp
```

## How it works

```
paste a link ──► yt-dlp ──┐   (only when the source is a URL)
                          ▼
ffmpeg  ──►  16 kHz mono WAV  ──┬──►  pyannote  ──►  who spoke when
                                │      (.venv-diar, torch cu13)
                                │
                                └──►  faster-whisper  ──►  words + timestamps
                                       (.venv, ctranslate2 cu12)
                                                  │
                                                  ▼
                                    word-level alignment ──► Markdown / JSON / SRT / VTT
```

Whisper segments routinely straddle a speaker change, so attribution happens at the **word**
level: each word goes to the diarization turn it overlaps most, short flickers ("mhm", "ja")
get absorbed into the surrounding speaker, and the result is re-grouped into turns.

### Why two virtualenvs

This is forced, not a preference. `ctranslate2` (faster-whisper's engine) requires
`libcublas.so.12`; PyPI `torch 2.13` is a **CUDA 13** build shipping `libcublas.so.13`. They
cannot live in one environment.

It also turns out to be the right architecture anyway: the two GPU stages run as **sequential
subprocesses**, so peak VRAM is `max(diarize, transcribe)` rather than their sum, and the CUDA
context is genuinely returned to the driver in between. That is what makes this fit on a 12 GB
card that is also hosting Ollama and the wisprflow daemon.

Two traps worth knowing if you touch the launcher:

- `sinribe-run` **must** put the cu12 wheels on `LD_LIBRARY_PATH`. Without it the whisper model
  constructs fine and then throws `Library libcublas.so.12 is not found` on the first encode.
- The diarization worker **must** run with `LD_LIBRARY_PATH` stripped. cuDNN 12 and cuDNN 13
  wheels share the SONAME `libcudnn.so.9`, so an inherited path makes torch load a cuDNN built
  against the wrong CUDA major.

## Install

pyannote's models are gated, so there is a one-time online step:

1. Create a HuggingFace account: <https://huggingface.co/join>
2. Accept the licence at <https://huggingface.co/pyannote/segmentation-3.0>
3. Accept the licence at <https://huggingface.co/pyannote/speaker-diarization-community-1>
4. Make a **Read** token at <https://huggingface.co/settings/tokens>
5. `./install.sh --hf-token hf_xxxxxxxx`

After that everything runs with `HF_HUB_OFFLINE=1`. The token is stored in
`~/.config/sinribe/hf_token` (chmod 600) and is never written into this repo.

`speaker-diarization-3.1` is offered as an alternative in the Options panel but needs its own
licence acceptance; `community-1` is newer and more accurate, so there is rarely a reason to.

## Performance

Measured on an RTX 5070 Ti Laptop (12 GB, sm_120) with the GPU already shared with Ollama
(~2.9 GB) and the wisprflow daemon (~2 GB):

| Input | Wall clock | |
|---|---|---|
| 43 s, 3 speakers | 8 s | |
| **4 h 00 m, 3 speakers** | **10 m 58 s** | **21.9× realtime** |
| same 4 h job, resumed from cache | 1.3 s | identical output |

The 4-hour run split as ~7 m 51 s diarization and ~3 m transcription. Peak RSS 8.7 GB; peak VRAM
11.3 GiB of 12.2 GiB — that last figure is why the tier thresholds below carry a GB of headroom.

Precision and batch size are chosen from free VRAM at launch, from `float16`/batch 16 (~9.1 GB,
measured) down through `int8_float16` to CPU `int8`. A CUDA OOM retries one tier smaller rather
than failing, but the thresholds are deliberately conservative: discovering an OOM eight minutes
into a four-hour job and redoing that stage is far more expensive than running one tier down.

Speaker attribution was verified against `tests/fixtures/three_speakers.truth.json` — 9 of 9
ground-truth spans correctly attributed, and identity stayed stable across the full 4 hours
(the same person is still Person 1 at 03:59 as at 00:00), which is the part that actually gets
hard at length.

## Accuracy

A 66-minute German interview, transcribed by Sinribe, by [Vibe](https://github.com/thewh1teagle/vibe),
and by stock `faster-whisper large-v3`, then scored word-by-word against a human transcript:

| | Sinribe | Claude (large-v3) | Vibe |
|---|---|---|---|
| Word error rate | **18.84%** | 20.17% | 20.51% |
| Meaning-changing errors | **7.46%** | 8.10% | 8.20% |
| Median timestamp error | **1.3 s** | 2.0 s | 14.0 s |
| Speaker attribution | **99.60%** | — | — |

Sinribe leads on every axis measured. Speaker attribution is scored against a 90.12% majority-class
baseline, and holds 97.7% recall on the interviewer even though they speak only 11% of the hour.

**[Full report, method and caveats →](https://darian-gajgic.github.io/Sinribe/benchmark/)** ·
[source and scoring code](docs/benchmark/)

## Resuming and re-exporting

Each job caches its decoded audio, diarization and transcription under
`~/.cache/sinribe/jobs/<hash>/`. Cancel a four-hour run and start it again — completed stages are
reused instead of recomputed. Checkpoints older than 14 days are purged at startup.

The sidecar `.json` holds word-level timestamps, so renaming a speaker (`Person 1` →
`Prof. Müller`) and re-exporting takes milliseconds and never touches the GPU.

## Options

| Option | Default | Notes |
|---|---|---|
| Model | `large-v3` | `medium.en` and `small` are also cached locally |
| Language | Auto-detect | Or pin it, which is slightly faster and safer on quiet openings |
| Speakers | Auto-detect | Or set exactly N, or a min–max range |
| Sidecar `.json` | on | Needed for instant re-export |
| `.srt` / `.vtt` | on | Speaker-prefixed cues |
| LLM chapters + summary | off | Uses `gemma3:4b` on your local Ollama (`:11435`); still offline |
| `yt_cookies_from_browser` | `""` | Config-file only. Set to `"firefox"` etc. for gated episodes |
| `keep_downloads` | `true` | Set false to expire cached downloads after `download_cache_days` |

Settings live in `~/.config/sinribe/config.json`.

Downloaded podcasts are large and are kept indefinitely by default. To reclaim space, either
delete `~/.cache/sinribe/downloads/` or set `"keep_downloads": false`.

## Tests

```bash
./.venv/bin/python -m pytest tests/ -q
```

`tests/fixtures/three_speakers.wav` is a synthetic 3-speaker fixture (two Piper voices plus a JFK
archive clip) with ground-truth spans in `three_speakers.truth.json`, so speaker attribution can
be checked against a known answer rather than by eye.

## Layout

```
sinribe-run              launcher (sets the CUDA-12 loader path)
install.sh               builds both venvs, fetches models, installs the .desktop entry
sinribe/
  pipeline.py            orchestrator: stages, checkpoints, progress, cancellation
  merge.py               word -> speaker alignment
  fetch.py               yt-dlp download + metadata probe (the only networked module)
  audio.py               ffmpeg probe/decode/clip
  enrich.py              optional Ollama chapters + summary
  textfmt.py             sentence splitting, timestamp formatting
  workers/asr_worker.py  faster-whisper subprocess   (.venv)
  workers/diar_worker.py pyannote subprocess         (.venv-diar)
  render/                markdown, subtitles, sidecar
  ui/                    PySide6 window, worker thread, speaker panel
```

## Credit where it is due

Reuses hard-won pieces from earlier tools on this machine: the CUDA loader shim and the EN/DE
sentence splitter from `local-wisprflow`, the colour palette and config handling from `Sinlate`,
and the "one model per subprocess, VRAM is freed by process exit" lesson from `Nexus`.
