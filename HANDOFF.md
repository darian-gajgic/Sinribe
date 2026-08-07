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
| Tests / lint | ✅ 119 pytest, ruff clean |
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
a room (`~/Sinribe-Transcripts/Kev/`), with a hand-made reference the user estimates at ~95 %
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
4. **More voting passes.** Per-word voting is the one mechanism that *does* work — but only for
   about three passes. The full ladder, measured 2026-08-07: 1 → 21.3 %, 2 → 20.8 %, 3 → 20.5 %,
   4 → 20.3 %, 5 → 20.5 %, 6 → 20.6 %, 7 → 20.6 %, 8 → 20.7 %, 9 → 20.6 %. The 1× rung therefore
   costs **31 minutes against about 6 for the ★ rung and ends up 0.2 pp worse**. Voting works by
   letting passes that mishear *different* words outvote each other; passes 5-9 are the same two
   models re-filtered and re-beamed, so they mishear the *same* words and the duplicates win the
   majority. Adding a genuinely third model would be the only thing worth trying here.

   **Measuring the whole ladder is nearly free and nobody should re-run nine decodes to do it.**
   One 1× run leaves `asr-pass0..8.json` in `~/.cache/sinribe/jobs/<key>/`, and
   `rover.combine(passes[:n])` on each prefix *is* rung n, because every rung's `extra_passes` is
   a prefix of the 1× rung's. All nine points score in seconds, on the CPU, from one run.

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
  textfmt.py             EN/DE sentence splitting, timestamp formatting
  workers/asr_worker.py  faster-whisper subprocess   (.venv,      cu12)
  workers/diar_worker.py pyannote subprocess         (.venv-diar, cu13)
  workers/refine_worker.py per-sentence voice-print check (.venv-diar, embedding model only)
  render/                markdown, subtitles, sidecar
  ui/                    PySide6 window, QThread wrappers, speaker panel
```

State on disk: config `~/.config/sinribe/config.json`; HF token `~/.config/sinribe/hf_token`
(chmod 600, never in the repo); stage checkpoints `~/.cache/sinribe/jobs/<hash>/`; downloaded
media `~/.cache/sinribe/downloads/` (kept indefinitely by default — first place to look when
disk gets tight).

## Design notes worth knowing before changing anything

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
   reference are in `~/Sinribe-Transcripts/Kev/` and a full sweep variant costs about five
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
