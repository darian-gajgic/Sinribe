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
| Summary + HTML presentation, Claude Code provider (default) | ✅ no API key; `claude -p` with the login already on this box |
| Summary + HTML presentation, local Ollama provider | ✅ end-to-end on 25 min of the Kokotajlo interview |
| Summary + HTML presentation, direct Claude API provider | ⚠️ request shape verified against a stub server; **never run against the real API, no key on this box** |
| Web research pass | ✅ via Claude Code: 6 turns, real sources, no permission denials |
| **Claude Code from the app menu** (desktop PATH has no `~/.npm-global/bin`) | ✅ fixed 2026-10-08, see "The CLI was invisible from the app menu" |
| **Claim-by-claim web check** (outdated / incorrect / disputed / open / confirmed / unverified) | ✅ 2026-10-08, see "The web check" |
| **New brief end to end**, 10m45s video, desktop PATH, Claude Code + web check | ✅ 2026-10-08: 5m 24s, **$3.25**, 3 sections, 14 claims (11 confirmed, 3 unverified), 148 tool sources, 3 min read |
| New brief on a 2 h podcast | ❌ not yet run; the $17.97 figure above predates the rework and the word budget |
| **Full summary of a 2 h podcast, Claude Code + research** | ✅ 2026-09-16: 12 sections, 72 blocks, 36 quotes, 7 tables, 17 verified links, 104 kB page. **18 minutes, $17.97 of usage** |
| Tests / lint | ✅ 300+ pytest, ruff clean |
| **German 34 min phone interview end-to-end** | ✅ 2026-07-26, 21.5× realtime |
| Voice-print speaker refinement | ✅ cut question+answer blocks from 28 to 11 on that file |
| **Word accuracy measured against a human reference** | ✅ 2026-08-06, see below |
| **A real multi-hour podcast *download*** | ❌ longest real fetch was ~10 min |
| `pyannote/speaker-diarization-3.1` pipeline | ❌ licence never accepted, so never downloaded |

## Accuracy: what is measured, and how to measure it

`sinribe/eval/` scores a transcript against a human reference. Use it before and after **any**
change to decode settings — this project has now twice adopted a change that sounded obviously
right and measurably made accuracy worse.

```bash
./sinribe-eval reference.docx transcript.md                          # grade one run
./sinribe-eval --sweep variants.json audio.wav reference.docx -o r.json   # compare settings
```

The benchmark file is a 65-minute Bavarian-accented German interview recorded on an iPhone across
a room (kept outside the repo, since it is a private conversation), with a hand-made reference the user estimates at ~95 %
accurate. It is deliberately hard: the two speakers score nine points apart.

**Baseline, `large-v3` as shipped 2026-07-26:** WER **21.3 %**, 87.0 % of words correct,
speaker attribution **98.9 %**. Per speaker: interviewer 93.4 % correct, interviewee (Bavarian,
90 % of the talk) 86.3 %. Error rate climbs 13.7 % → 38.8 % across the hour as the conversation
speeds up; the audio itself is uniform throughout (−19.1 LUFS, no clipping, no rumble), so that is
speech, not recording quality.

### How to time anything here without fooling yourself

Two separate wrong conclusions in this work came from bad timing, so read this before quoting a
speed. **WER is reproducible to the byte; wall clock is not.**

- **Never run anything else on the GPU during a sweep.** An early sweep reported beam 8 as
  *faster* than beam 5, which is impossible — the variants were competing with test runs of mine.
  It is shared with Ollama and whatever else is resident, so check `nvidia-smi` first.
- **Interleave, do not block.** Running all the sequential variants and then all the batched ones
  lets drift masquerade as a difference. A/B/A/B settles it: sequential measured 193 s and 194 s,
  batched 77 s and 77 s, in one run.
- **Watch for cold checkpoints.** One batched run measured 197 s against another's 77 s with
  identical settings; the difference was 120 s of diarization it had to recompute. `sweep.py` now
  detects and flags this, but check the ⚠ column before believing a number.

**Thirteen configurations were scored. The baseline won.** The full frontier — the only settings
not beaten outright on both speed and accuracy — is three points, and it is what `presets.RUNGS`
now contains:

| | WER | RTF | |
|---|---|---|---|
| `large-v3` sequential, beam 5 | **21.3 %** | 12.6× | the default, unbeaten on accuracy |
| `large-v3` batched, beam 5 | 21.6 % | 50.4× | 4× faster for ~1 word in 100 |
| `large-v3-turbo-german` batched | 23.1 % | 84.9× | for finding your way around a long file |

What the diagnosis ruled *out* is as useful as what it ruled in:

- **Coverage is not the problem.** Deletions are 4.4 % and 165 of 177 runs are one or two words.
  The recogniser hears something almost everywhere; it hears it wrong. Substitutions plus
  insertions are the whole 21 %.
- **Diarization, merge and the voice-print refiner are not the problem.** 98.9 % of correctly
  transcribed words are attributed to the right person. Leave them alone.
- **Confidence gating cannot be the main fix.** Wrong words come back with a *median* probability
  of 0.86. Re-decoding everything under 0.7 would cover 10.6 % of the audio and catch a third of
  the errors. That band is still worth surfacing to a human (see `review_spans`), but it is not a
  correction mechanism.
- Roughly one substitution run in eight is only a spelling difference (`gernhabt`/`gern habt`).
  The scorer counts those as correct, or every real measurement drowns in them.

Scoring notes that took a while to get right, and should not be undone:

- Alignment is **sequence-based, never time-based**. The reference is two recordings stitched
  together with timestamps that restart mid-document; matching on word order made that a non-issue
  instead of a data-repair job.
- The reference's `(unverständlich)` markers are dropped, not scored. They mark audio the human
  could not resolve either.
- Speaker mapping is **one-to-one**. Majority-vote mapping gives a diarizer that lumped everyone
  together a perfect score, because every reference speaker maps to the single label it emitted.
- Absolute WER is slightly pessimistic here: some "insertions" are passages the human skipped
  (one is 28 words of real speech). Relative comparisons between settings are unaffected, which is
  what the sweep is for. The "how much of what the human wrote did it get wrong" figure is
  substitutions + deletions, 13.0 % on the baseline.

### Things that sounded right and measured worse

Recorded so they are not re-tried. Each cost one sweep run to disprove and would have cost a lot
more to discover in use.

| Change | Expectation | Measured |
|---|---|---|
| `condition_on_previous_text=True` | Whisper's LM context fixes function-word errors, which dominate | **21.3 % → 24.4 %.** Deletions 395 → 749, attribution 98.9 % → 94.2 % |
| `large-v3-german` (primeline fine-tune) | German-only training beats multilingual on German | **21.3 % → 24.8 %.** More substitutions *and* more insertions |
| `large-v3-german` + context | Both fixes together | **99.2 % — the decode collapsed**, see below |
| `hotwords` with the interview's proper nouns | The documented fix for mangled names | **21.3 % → 22.3 %.** "Dieter" recovered; *Fujitsu* still "Jitze"/"FIUZI", *VR-Bank* still wrong |
| Wider beam (8) with `patience=2` | More search on hard audio | 21.7 %, and slower |
| More VAD padding (700 ms / 300 ms silence) | Catch the quiet run-ins that scored worst | 21.7 % |
| `vad_filter=False` | Stop the VAD trimming speech | 22.7 % |
| `audio_filter="clean"` (highpass + level) | Help the voice further from the phone | 21.5 % — a wash |
| `audio_filter="denoise"` (+ FFT denoise) | Same, harder | 22.3 % — denoising removes signal whisper uses |
| `temperature_fallback=False` | Stop hard windows falling back from beam search to sampling | 21.3 % — **byte-identical output**; the fallback never fires here |

The temperature-fallback row is the useful kind of null result: the transcript bodies from that run
and the baseline are byte-identical, so whisper's fallback ladder is not firing on this recording
at all and cannot be responsible for anything. It also re-confirms reproducibility — two runs,
days-of-work apart, same bytes.

The audio front-end (`audio.FILTER_CHAINS`, wired through `decode.audio_filter`) stays in the code
but is off everywhere and no rung enables it. It was worth testing and it lost: this recording is
technically clean — −19.1 LUFS integrated, 2 clipped samples in 189 million, no rumble, uniform
level across all 65 minutes — so there was nothing for a filter to fix, and the denoiser took away
signal whisper was using. Reach for it only on a recording whose *measurements* are bad, and
measure the result.

Hotwords are worth keeping as a feature — they are the supported mechanism, they cost nothing, and
on a recording where the names are merely unfamiliar rather than badly heard they will help. But
they are a vocabulary bias, not a correction: against a word the acoustic model confidently hears
as something else, a prompt hint loses. The feature ships with that written on it, in the tooltip
and the README, rather than as a promise.

The German fine-tune is the interesting one. Its published numbers are real, but they are on
Common Voice — *read* speech. Fine-tuning onto a narrower, cleaner distribution appears to cost
large-v3's robustness on far-field spontaneous dialect, which is precisely this recording. It also
decoded twice as fast (23.7× vs 12.5×) because it hits far fewer temperature fallbacks: it is more
confident, not more correct. It is still installed and selectable, and may well win on clean
close-miked German — but it is not the default, and "use the German model" is not advice to give
without measuring the specific recording.

### `large-v3-german` returns an empty transcript when given hotwords

Not a subtle degradation — zero words, for the whole recording, with a hotword string as short as
a single letter. Reproduced on two different files. The same model with no hotwords transcribes
normally, and `large-v3-turbo-german` from the same publisher is fine, so this is that particular
fine-tune having lost the ability to condition on a prompt prefix. Its `tokenizer.json` and
`vocabulary.json` are byte-identical in size and structure to the base model's, so it is not a
conversion fault and there is nothing to fix locally.

`config.MODELS_WITHOUT_HOTWORDS` records it. The pipeline drops the hotwords and logs loudly
rather than failing the job, and the UI disables the field when that model is selected so the
combination cannot be built by accident. If a future model shows the same behaviour, add it to
that set — the coverage guard below will catch it first.

### The "Worth a listen" list, and why it is no longer capped

`render/markdown.review_spans()` surfaces the passages whose word probabilities sit below 0.5.
Measured on the benchmark: that band is 483 of 9 399 words, **57 % of them genuinely wrong**, and
it catches 18 % of all errors. Lowering the threshold trades precision for recall almost linearly
(0.7 → 51 % precision, 33 % recall), so 0.5 is a judgement call, not an optimum.

The list shipped capped at 60 entries on the theory that nobody would work through more. That was
the wrong call: the benchmark interview flags **363 passages**, so the cap silently withheld 303
places worth an ear from someone whose whole purpose in opening the section was to find them. A
checklist that stops at a sixth of the job is worse than a long one. `max_spans` now defaults to
0, meaning all of them, in time order; `review_max_spans` in the config sets a positive cap for
anyone who wants a sample instead.

Ordering still matters when a cap *is* set: the list keeps the *least confident* entries and only
then restores time order. An earlier version truncated in time order and silently handed back a
review list that stopped at minute ten of sixty-five.

This is the part of the accuracy work with the clearest payoff. It does not lower WER at all; it
lowers how long a human spends finding the errors that remain, which given the 8.7 % floor is
where the actual leverage is.

### Whisper can collapse silently — there is now a guard

`large-v3-german` with context conditioning returned **156 words for a 65-minute interview** and
exited 0. No error, no warning, a clean-looking Markdown file that stops at 00:26. Nothing in the
pipeline noticed, because nothing was comparing the transcript to the audio.

`pipeline.speech_coverage()` now does, using the diarization as an independent second opinion:
diarized speech time versus the time actually spanned by words. Healthy runs on the benchmark
score 100 %; the collapsed one scores 0.8 %. Below 50 % the job logs a warning and the transcript
carries a visible banner. The threshold is deliberately far from the healthy band — this is a
"something is catastrophically wrong" detector, not a quality metric.

### There is no "spend more compute, get better accuracy" setting. This was chased properly.

The obvious product request — a quality slider whose slow end is genuinely more accurate — has no
implementation with these models. Three separate mechanisms were measured and all of them failed:

1. **Wider beam search.** Beam 20 with patience 2 costs **5.6× the compute** (3.8× realtime
   against 21.2×, seventeen minutes instead of three) and returns 86.9 % of words correct against
   the default's 87.0 %. Thirteen words out of 9 025, in the wrong direction.
   The reason is worth internalising: beam search explores *the model's own hypothesis space*.
   It helps when the right answer was in the model's head and the search missed it. Here the
   errors are acoustic — the model's probability distribution is itself wrong about whether it
   heard "das" or "es" — so searching a wrong distribution harder returns the same wrong answer
   at five times the price. Every "spend more time" knob on this list fails for the same reason:
   they are all more search over unchanged acoustics.
2. **Combining several different decodes.** A confidence-weighted picker over four diverse
   transcripts (default / German model / clean filter / wider VAD), choosing per time window
   whichever decode was most confident, scores **21.9–22.1 % against 21.3 % for the best single
   decode** — worse, at four times the cost, at every window size from 5 s to 20 s. It fails
   because whisper's confidence barely tracks correctness (median 0.86 on wrong words), so the
   picker confidently selects the worst of the four more often than any other.
3. **A perfect oracle picker** — the theoretical ceiling for (2) — reaches 89.5 % over two models
   and 91.3 % over five, against 87.0 % for large-v3 alone. So even flawless selection is worth
   ~4 points, and the realistic version is worth less than nothing. Check this with
   `sinribe-eval --ceiling` before anyone proposes an ensemble again.
4. **More voting passes.** Per-word voting is the one mechanism that *does* work. Through three
   passes it is measured and holds: 1 → 21.3 %, 2 → 20.8 %, 3 → 20.5 %.

   **Past three passes the 2026-08-07 ladder is void — it measured a bug, not a setting.**
   `pipeline._decode` read `audio_filter` from the job config instead of the pass's, and only the
   extra passes ever set one, so the "denoise" and "clean" passes decoded the *untouched* WAV and
   returned the pivot's own words back into the vote. Pass 5 was a copy of pass 1 and pass 6 was
   another. That is why the curve looked like it turned back up (4 → 20.3 %, 5 → 20.5 %,
   6 → 20.6 %, …): the pivot was casting two or three of the votes. "The extra passes duplicate a
   model that has already voted" was a description of the defect, not a property of voting.
   Fixed 2026-08-07 (each pass now gets the audio its own settings ask for, with a regression
   test in `tests/test_pipeline.py::TestVotingPlan`); the 5×, 2× and 1× rungs are now
   **uncalibrated** and their WER figures have been pulled from `presets.py` until a sweep
   re-earns them. Note that 4 passes scored **20.3 %** — the best point on the whole ladder, and
   the last one before the first duplicate — so "five real passes beat three" is an open question,
   not a settled no.

   Two guards were added with the fix: a pass that resolves to the same model *and* the same
   decode settings as an earlier one is dropped with a log line (pinning a model in the dropdown
   used to turn the "different model" pass into a silent re-run of the pivot), and a pass that
   cannot get the audio it asks for is skipped rather than run on the wrong audio.

   **Measuring the whole ladder is nearly free and nobody should re-run nine decodes to do it.**
   One 1× run leaves `asr-pass0..8.json` in `~/.cache/sinribe/jobs/<key>/`, and
   `rover.combine(passes[:n])` on each prefix *is* rung n, because every rung's `extra_passes` is
   a prefix of the 1× rung's. All nine points score in seconds, on the CPU, from one run.
   This is also how the bug above was invisible for a day: the prefix trick faithfully reproduced
   what the shipped code did, and what the shipped code did was vote on two copies of pass 1.
   **Before trusting a ladder measured this way, check that the passes actually differ** —
   `[w["word"] for w in json.load(open(f))["words"]]` for each cached pass, and no two lists may
   be equal.

The floor is 8.7 % of words that no configuration gets right. Accept it, and spend the effort on
the review workflow instead.

### Progress reporting has to account for voting

Each pass reports its own 0..1. Sent straight to the bar, a 1× run filled it and snapped back to
the start **nine times**, while the ETA — elapsed ÷ fraction done — announced the job was nearly
over once per pass. `pipeline.Runner._subspan()` confines each pass to its own slice of the
transcribe stage and `_frac()` derives the estimate from progress through the whole *stage*, not
the current pass. The slices are weighted by `_pass_cost()`, because the passes are not remotely
equal: on the benchmark run a batched pass took one minute and the beam-20 pass took ten, so equal
slices would still have looked stuck for a third of the job. Anything added to `extra_passes` in
future needs a cost there too.

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
sinribe interview.wav -q 4 --hotwords 'Fujitsu, VR-Bank'   # slower rung + expected names
./sinribe-eval reference.docx transcript.md         # score it against a human transcript
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
sinribe-eval             the scorer / sweep runner, same loader path
install.sh               builds both venvs, fetches models, installs .desktop  (--skip-venvs)
tools/
  convert_german_models.sh  CTranslate2 conversion in a THROWAWAY venv (never touch .venv)
sinribe/
  presets.py             the quality ladder: each rung's decode settings + its measured WER
  eval/                  accuracy measurement — nothing here runs during a normal transcription
    reference.py           load .docx/.odt/.md/.srt, incl. multi-part restarting timestamps
    normalize.py           German-aware tokenisation; spelling variants must not count as errors
    align.py               difflib anchors + edit-distance DP with free compound merges
    score.py               WER, per speaker, over time, speaker attribution, confusion pairs
    sweep.py               decode N ways, score each, one table
    ceiling.py             how much a perfect chooser would win — the "keep tuning?" answer
  pipeline.py            orchestrator: stages, progress weights, checkpoints, cancel, VRAM tiers
  merge.py               word -> speaker alignment  ← accuracy lives here
  fetch.py               yt-dlp download + probe    ← the ONLY networked module
  audio.py               ffmpeg probe/decode/clip
  enrich.py              optional Ollama chapters + summary
  summary/               the briefing feature — a second job that reads the finished transcript
    providers.py           Claude API (official SDK) and Ollama (raw urllib) behind one interface
    build.py               the passes: notes -> plan -> research -> sections -> close
    schema.py              JSON Schema for each pass + the clamping that makes answers renderable
  textfmt.py             EN/DE sentence splitting, timestamp formatting
  workers/asr_worker.py  faster-whisper subprocess   (.venv,      cu12)
  workers/diar_worker.py pyannote subprocess         (.venv-diar, cu13)
  workers/refine_worker.py per-sentence voice-print check (.venv-diar, embedding model only)
  render/                markdown, subtitles, sidecar
    webpage.py             the summary page: inline CSS, no JS, no external requests
  ui/                    PySide6 window, QThread wrappers, speaker panel
```

State on disk: config `~/.config/sinribe/config.json`; HF token `~/.config/sinribe/hf_token`
and the optional Claude key `~/.config/sinribe/anthropic_key` (both chmod 600, never in the repo); stage checkpoints `~/.cache/sinribe/jobs/<hash>/`; downloaded
media `~/.cache/sinribe/downloads/` (kept indefinitely by default — first place to look when
disk gets tight).

## Design notes worth knowing before changing anything

### The summary needs no API key, and that is deliberate

Three providers sit behind one interface (`summary/providers.py`), and `auto` tries them in this
order: **Claude Code**, then the direct API, then local Ollama.

Claude Code shells out to `claude -p --output-format json`. It authenticates with the claude.ai
login already on the machine, which is the whole point: the hand-built page that set the format
for this feature was written that way, so requiring a separate API key would have been a
regression dressed as an integration. Verified working here; `claude auth status` is the probe
(a local read, no billed request).

What it gives up is decoder-enforced structured output. There is no `output_config.format` over
the CLI, so the schema is appended to the *user* turn and the answer is parsed hopefully rather
than guaranteed. In practice Opus returns clean JSON; `providers.salvage_json` and `_unfence`
cover a fence or a truncation. The schema must stay in the user turn, not the system prompt: a
resumed session keeps the system prompt it was created with, so every section after the first
would otherwise be asked for the plan's shape.

What it gains is a real conversation. One job is ONE resumed session (`--resume <session_id>`):
the transcript goes out once and each later pass continues it, so the transcript is read from
cache (measured: 2 input tokens against 11,677 cache reads on a resumed turn) and each section is
written with the earlier sections in view. That beats the API path's identical-cached-prefix
approach, where sections are independent and can restate each other.

Hardening, because this runs a subprocess that can reach the network: `--restricted` (no Bash, no
code execution, no WebFetch unless named, user settings ignored, file tools confined),
`--strict-mcp-config` (none of the user's MCP servers or claude.ai connectors load; verified,
`mcp_servers: []` in the init event), `--disable-slash-commands`, a scratch cwd under
`~/.cache/sinribe/` so no project `CLAUDE.md` is inherited (verified: a print-mode run from such a
directory reports no CLAUDE.md and no hooks), and the prompt on stdin rather than argv. The JSON
passes get `--tools ""`, no tools at all. Only the research pass gets `--tools WebSearch WebFetch
--allowedTools WebSearch WebFetch --permission-mode dontAsk --no-session-persistence`, in
`stream-json` so its sources can be read from the tool results. Anything else it reaches for is
denied and the denial is logged. `--effort` passes `summary_effort` through (default high).

**The CLI updates itself, and npm replaces the symlink to do it.** This cost a real run: the
plan pass finished, `claude` auto-updated 70 seconds into the job (symlink and package directory
both restamped), and the research pass died with "the `claude` command is not on PATH" from a
binary that was there a minute earlier and is there now. Three defences, all in
`ClaudeCodeProvider`: `DISABLE_AUTOUPDATER=1` in the subprocess environment (our subprocess only,
which also keeps one job on one version), the resolved path cached after the first lookup instead
of re-running `which` fifteen times, and a few one-second retries before declaring it absent.
Worth remembering if anything else here ever shells out to `claude`.

**What it costs, measured, because the number is larger than it looks.** One full run over the
139 kB Kokotajlo transcript with research on: **$17.97 and 18 minutes**. That is 15 Opus calls,
and the resumed session is part of why: every section turn re-reads the whole conversation so far
(transcript plus all earlier sections), which grows with each section. Coherence is bought with
tokens. The levers, in the order worth trying: turn research off, lower `summary_max_sections`
(12 is the ceiling and the cost is roughly linear in it), or set `summary_model` to
`claude-sonnet-5`. My earlier note that resuming is "strictly better" than the API path's
independent cached prefixes was wrong: it is better for coherence and worse for cost.

### The CLI was invisible from the app menu, and "auto" hid it

Found 2026-10-08, after a brief came out written by `gemma3:4b` while the user believed Claude
was writing it. `npm install -g` with a user prefix puts `claude` in `~/.npm-global/bin`, and
that directory is added to PATH in `~/.bashrc`, which only interactive shells read. The desktop
session's PATH (checked on the running `gnome-shell`: `~/.local/bin:/usr/local/bin:/usr/bin:...`)
does not have it, so a Sinribe started from the app menu ran `shutil.which("claude")`, found
nothing, and either refused the explicit `claude-code` choice or let `auto` fall through to
Ollama without saying so anywhere the user would look.

Fixed in three places:

- `ClaudeCodeProvider._locate` checks PATH first, then `CLAUDE_CODE_CANDIDATES` (npm-global,
  `~/.local/bin`, the native installer's `~/.claude/local`, bun, volta, `/usr/local/bin`,
  Homebrew) and nvm's per-version bins. `claude_binary` may also be an absolute path. The binary
  here is a native ELF (`claude.exe` inside the npm package), so no `node` on PATH is needed.
- `pick()` records what `auto` skipped and why on `provider.skipped`. The pipeline logs it, the
  GUI's provider label says "(fallback, see tooltip)" and asks before starting, and the page
  footer states it.
- Asking for web research with a provider that cannot search is now a setup error at second
  zero, not a log line and an unchecked page that looks exactly like a checked one.

### The web check

Rebuilt 2026-10-08 around the question the user actually has: "this video is two months old and
about AI; which of its facts no longer hold?" The old pass searched once against the outline,
wrote prose, and its links were trusted if they appeared in that prose.

- **Claims come from the plan pass** (`PLAN_WITH_CLAIMS_SCHEMA`), which has just read the whole
  transcript: up to `summary_max_claims` (16), time-sensitive first, each with timestamp and
  section. The research call never sees the transcript.
- **The publication date reaches every prompt.** `fetch.MediaMeta.published` comes from
  yt-dlp's `release_date`/`upload_date`/`timestamp`; the source block says "Published on:
  2026-08-03 (66 days before today)". Without it "outdated" had nothing to be relative to. The
  uploader's description and chapter markers go to the plan and research prompts too, and links
  in the description are allowed as sources.
- **Research then verify, two calls.** `provider.research()` has WebSearch and WebFetch
  (`--restricted` drops WebFetch unless `--tools` names it) and writes a report; a tool-free
  `ask_json(VERDICTS_SCHEMA)` turns it into verdicts. Statuses: outdated, incorrect, disputed,
  open (a prediction not yet due), confirmed, unverified.
- **Sources are what the tools returned.** Claude Code research runs with `--output-format
  stream-json --verbose`; `_stream_envelope` collects `tool_use_result` URLs (WebSearch hits,
  WebFetch pages that came back 2xx/3xx). The API path walks `web_search_tool_result`,
  `web_fetch_tool_result` and text citations. A verdict other than unverified with no such
  source is downgraded to unverified; a development with none is dropped.
- **Every link in the brief is held to that list** (`schema.restrict_links`), not only the
  Sources section. Before this, `[label](url)` inside section prose went straight to the page.
- **A failed web check no longer loses the page.** It used to be an unguarded call; a CLI
  timeout there raised out of `summarize()`. Now it is caught, the brief is written from the
  transcript, and the top of the page says the check did not run.
- **Where it shows up:** freshness banner with verdict counts, a heads-up inside the 1-minute
  version, a "What has changed" section, an amber callout at the top of each affected section,
  and the section writer is told to state both what was said and what is true now.

**The full summary has a word budget** (`build.section_budget`): 15 % of the spoken words,
clamped to 400-3,000 in total and 150-350 per section. Measured without it on a 10m45s video
(2,400 spoken words): the sections came to about 2,900 words, longer than the transcript, and
the page advertised "13 min full summary" next to "10m 45s listening". The one-minute version was
tightened at the same time (gist 50-80 words, 3-5 one-sentence takeaways, about 200 words in all,
the reading speed of a non-native reader of dense English), and the "instead of N listening" pill
only appears when it is true.

The brief itself was restructured for learning at the same time (1-minute version, key terms
before the detail, prose-first sections with key points, worth-watching moments, a self-test).
The six forecasting lenses copied from the Kokotajlo page were replaced by optional boxes that fit
any recording: example, evidence, forecast, caveat, howto, definition.

### The summary is a second job, and it fails differently from the first one

`enrich.py` swallows every error and returns `{}` — right for a stage nobody asked for. The
summary is the opposite: the user ticked a box for it, so it is loud.

- The provider is **probed before the audio is decoded**. A missing key raises `StageError(kind=
  "setup")` at second zero instead of after two hours. The GUI probes again at Start so the usual
  case is a dialog, not a failed job.
- The probe checks that a credential actually **resolved**, not that a client constructed.
  `anthropic.Anthropic()` with nothing configured returns a working object and only raises at
  request time, so a construction-only probe reports "ready" on a machine with no key at all.
- After the transcript is on disk, the summary can no longer fail the job. It logs, sets
  `result["summary_error"]`, and the window says so in the done card.
- `elapsed` stays the transcription's time. The summary's is `summary_seconds`, reported
  separately, so the number printed in the transcript header still matches the file it is in.

### Seven failures real runs hit, and what fixed them

Every one was found by running it, not by reading it. Most are the local model's; the
mid-word clamp and the auto-updater are not. The lesson underneath the model ones:
grammar-constrained sampling guarantees the JSON is *valid so far*. It does not guarantee the
model reaches the closing brace, and a decoder that stops early produces output that is correct
and useless in equal measure.

1. **Section 2 truncated at char 5345.** `num_ctx` covers prompt *and* generation; 8192 minus a
   6 kB prompt left no room for a 4096-token answer. The window is now sized per call from the
   prompt actually being sent (`OllamaProvider._window`).
2. **The closing pass truncated at char 43.** The notes digest is unbounded in the number of
   chunks: 25 minutes of audio produced ~48 kB of notes, and four hours would produce ten times
   that. Fixed by capping each note (`_render_notes`) and folding batches of notes back through
   the model until they fit (`_digest`). Folding rather than truncating on purpose — truncation
   drops the *end* of the recording and yields a summary of the first half labelled as the whole.
3. **A section failing killed the page.** One `SummaryError` propagated out of the loop. Now a
   failed section is logged and skipped; only an empty page is an error.
4. **A truncated answer threw away the 90 % that arrived.** `providers.salvage_json` closes the
   open brackets and parses the complete prefix, dropping any half-written value. The schema layer
   already treats missing fields as empty, so a cut-off section arrives slightly shorter instead
   of not at all. Used on both providers: Claude can hit `max_tokens` too.
5. **Twelve sections asked of a 100-second clip.** The plan came back with no sections and the
   whole summary failed. Two fixes: the section count now scales with duration (`_section_count`,
   about one per ten minutes, floor 3, ceiling from config), and the plan — the one pass nothing
   can be dropped from — is retried once with a smaller ask (`_plan`).

6. **A clamped field ending mid-word.** A real page's "For you" paragraph ended "...would not
   press a button that s", because `_s` hard-sliced at its character limit. `schema._clip` now
   cuts at the last sentence boundary, or at a word boundary with an ellipsis, and the paragraph
   limits were raised where they were trimming content rather than padding.
7. **A field that swallowed the rest of the object.** gemma3:4b wrote the *typographic* quote
   inside a JSON string, so the string never closed where it meant to and one heading arrived
   carrying `", "subheading": "...", "focus": "..."`. The answer parses, so nothing upstream
   notices. `schema._JSON_LEAK` cuts a string at the first fragment that looks like one of these
   schemas' own keys reopening — restricted to those key names precisely so it cannot cut a real
   sentence — and the swallowed keys fall back to their defaults.

Worth keeping in mind when changing any of this: a bigger `max_tokens` is not the fix for a
truncated answer on the local path, because `num_ctx` has to grow with it. Ask for less, or give
the window more room, and prefer the first.

**`gemma3:4b` is the floor, not the recommendation.** It is what was already on the box, and the
five failures above are mostly its failures. It writes a complete page for 25 minutes of audio,
and falls over on a two-minute clip. Anything using the offline path seriously should pull a
12-14B model (`ollama pull qwen3:14b`) — there is VRAM for it — and `llm_model` in the config
already points wherever you want.

### En and em dashes are removed in code, not asked for in the prompt

`schema.dedash` runs over every generated string. The prompt asks the model to avoid them and a
small model obliges about half the time, which is not a guarantee. Between digits a dash is a
range and becomes a hyphen; spaced, it is doing a comma's job and becomes one; anything left
becomes a hyphen. The character class is U+2012..U+2015 plus U+2212, matching the
`grep -P '[\x{2012}-\x{2015}\x{2212}]|&[mn]dash;'` check, and a test asserts the rendered page
contains none.

### Citations are matched, never trusted

Every URL on the page is checked against URLs that appeared in a real search result or in the
transcript (`schema.close`, `schema.find_urls`). A link the model produced from memory is dropped.
This is the one place where a plausible wrong answer is actively harmful: an invented citation is
indistinguishable from a real one.

- **Attribution is sentence-level, decided from word timings.** Whisper *segments* straddle speaker
  changes, so they cannot be the unit. Individual *words* are too small in the other direction: one
  200 ms token against a boundary that is routinely 300 ms late tears a clause in half. So each
  word is assigned by maximal overlap, then every **sentence** takes its duration-weighted majority
  (`merge.vote_sentences`); runs shorter than 3 words flanked by the same speaker are still absorbed
  (otherwise transcripts ping-pong on every "mhm"). `sentence_atomic: false` restores word-level.
- **Diarization is a prior, not an oracle.** The `refine` stage (`workers/refine_worker.py`) builds
  one voice print per speaker from the diarization's long *isolated* turns, embeds each sentence's
  own audio, and overrules the label when the cosine gap to the runner-up clears `refine_margin`.
  This is the only thing that repairs a span the diarizer labelled outright wrong — regrouping
  cannot, because by then the wrong label is the only evidence left. Measured margins on real audio
  are bimodal (confident corrections 0.3+, coin flips under 0.1), which is why 0.15 is safe; spans
  under `refine_min_seconds` are never re-scored because a 0.3 s "Okay." has no voice in it.
  Sentences the refiner places are locked against flicker smoothing.
- The refine stage loads **only** the embedding half of the pipeline (~2 s, then 2.2 ms per crop —
  5 s total on a 34 min file) and is non-fatal by construction: any failure logs and leaves the
  plain diarization result in place. One speaker or no text and it exits before loading anything.
- **Decode settings live in `presets.py`, not scattered through the config.** The quality slider
  picks a rung; `presets.decode_settings(cfg)` resolves it, and `decode_overrides` lets the sweep
  drive one knob without inventing a config key per experiment. The old `asr_mode` / `beam_size`
  config keys are gone — `config._migrate` maps an existing install onto the equivalent rung.
- **`asr_mode: accurate`** decodes sequentially instead of batching. Batching splits at VAD
  boundaries and decodes the pieces independently, which is where edge words vanish. Measured
  head to head on the benchmark interview: batched **21.6 % WER / 85.9 % of words correct at
  50.4× realtime**, sequential **21.3 % / 87.0 % at 12.6×**. So sequential buys 1.1 points of
  words — almost entirely fewer deletions, 395 against 547 — for four times the wall clock. That
  is a real choice, not an obvious one, and it is what the quality slider now exposes.
  One thing that turned out **not** to be true: batching was assumed to loosen word timestamps
  enough to hurt speaker attribution. It does not — 99.0 % batched against 98.9 % sequential, and
  the interviewer's word accuracy is identical to the decimal. Don't avoid batching for that
  reason. Both modes are reproducible run to run.
- An `initial_prompt` full of German fillers was tried to push whisper toward verbatim output and
  **made word accuracy worse** ("selbstverständlich" for "selbstständig"). Don't re-add it.
  Whisper normalises disfluencies by design; no decode setting makes it a verbatim transcriber.
  `hotwords` is the supported way to bias vocabulary and is wired in — measured as a weak lever,
  see the table above.
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

0. **Before changing any decode setting, run the sweep.** Three of the four changes that looked
   most promising on paper measured worse, one of them catastrophically. The benchmark file and
   reference are kept outside the repo (ask the owner) and a full sweep variant costs about five
   minutes, because the decoded audio and the diarization are shared across variants.
   The honest summary of where accuracy stands: word errors on this recording are dominated by
   the 10.5 % of words that *no* configuration tried so far gets right — fast, broad-dialect,
   far-field speech. The remaining wins are in making correction cheap (the "Worth a listen"
   section) and in getting names right (hotwords), not in another decode knob.
1. Run a **real** 2–4 h podcast link end-to-end — the download half at that scale is the only
   untested part. Transcription at 4 h is already proven.
2. If speaker attribution still disappoints, look at the refine stage's log line first — it prints
   the **voice-print separation** (cosine between the speakers' prints). Around 0.2 means the
   voices are easy to tell apart and any remaining error is a merge/threshold problem; approaching
   0.6+ means they genuinely sound alike on this recording and no threshold will save it. After
   that the knobs are `refine_margin`, `refine_min_seconds`, `speaker_mode` (pin the count),
   `flicker_min_words` and `turn_gap_s`.
3. Attribution quality is measured, not eyeballed: on an interview, count how many rendered turns
   contain a '?' line followed by more text (question and answer collapsed into one block). It
   went 28 → 18 with sentence-atomic voting → 11 with voice prints on the 34 min German file.
