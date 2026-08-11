# Accuracy benchmark — German interview, 66 minutes

**[→ Read the full report](https://darian-gajgic.github.io/Sinribe/benchmark/)** (charts, per-section
analysis, ground-truth spot checks)

One 66-minute German coaching interview, transcribed three ways and scored word-by-word against a
human transcript. Same audio, same reference, same scoring code for every system.

| | Sinribe | Claude (large-v3) | Vibe |
|---|---|---|---|
| Word error rate | **18.84%** | 20.17% | 20.51% |
| Meaning-changing errors | **7.46%** (684) | 8.10% (743) | 8.20% (752) |
| Median timestamp error | **1.3 s** | 2.0 s | 14.0 s |
| Words within 2 s of truth | **61.5%** | 49.5% | 2.8% |
| Speaker attribution | **99.60%** | — | — |
| Segments | 160 turns | 1060 fragments | 929 fragments |

Sinribe leads on every axis measured.

## What was compared

- **Sinribe** — `large-v3` · CUDA float16 · diarization `speaker-diarization-community-1` · 3-pass vote.
- **Vibe** — transcript as supplied by the tool.
- **Claude (large-v3)** — a neutral baseline transcribed by Claude for this benchmark using stock
  `faster-whisper large-v3`, float16 on an RTX 5070 Ti, beam 5, VAD on, language forced to German.
  Single pass, no diarization, no post-processing. 65 min of audio in 139 s (28× realtime).
- **Reference** — a human-written transcript, 9,174 words after normalisation.

## Method

Exact word-level Levenshtein alignment with backtrace. Text is lower-cased, punctuation and quote
marks removed, umlauts preserved; stage directions and `Unverständlich` markers stripped. Identical
code path for all three systems — see [`scoring/`](scoring/).

Errors are additionally classified as **cosmetic** (meaning preserved — `gern`/`gerne`, a dropped
`ja`, a swapped article, or two spellings that are identical under German phonetics via Kölner
Phonetik) or **substantive** (the word a reader takes away changes — `Fieber → Führer`,
`Trance → Branche`). The substantive rate is the number worth optimising.

## Notable findings

**Vibe fabricates its opening.** Its transcript begins with 26 words that are not in the recording
(*„Also, sind wir doch safe! Ja, dann sage ich erstmal vielen Dank…"*). Verified by re-decoding the
first 25 seconds with voice-activity filtering disabled so nothing could be silently skipped: the
file starts mid-sentence on *„Ich frage nochmal ganz offiziell…"*, as both Sinribe and the human
transcriber have it.

**Vibe's timeline is ~14 s late throughout.** Measured across ~7,700 aligned word anchors, and
confirmed against the audio directly — a line Vibe stamps `30:01` is actually at `29:47`. The offset
holds near-constant from the first minute to the last, so it is systematic rather than accumulating
drift.

**Speaker attribution is the clearest differentiator.** Sinribe puts 99.60% of words on the right
speaker against a 90.12% majority-class baseline, and recovers 97.7% of the interviewer's words even
though they speak only 11% of the hour — the case that is easy to fail silently. Neither other
system emits speaker labels at all.

## Caveats, stated plainly

- **The baseline shares Sinribe's base model.** Both build on `large-v3`, so the gap between them
  measures Sinribe's pipeline — multi-pass voting, segmentation, diarization — not a different
  underlying model.
- **The human reference is not perfect.** It has two out-of-order timestamps near the end, so timing
  was measured only over its monotonic span (first 8,847 words). It also condenses speech in places,
  which charges every ASR system for words that were genuinely spoken. That penalty falls on all
  three equally, so the ranking holds, but it inflates every absolute WER figure here.
- **Speed and cost are not measured.** The Sinribe run used for this comparison was served from its
  cache, so no like-for-like throughput comparison was possible.
- **One recording, one language, two speakers.** These numbers describe this interview, not a claim
  about German ASR in general.

## Reproducing

The source audio and transcripts are not included here — the recording is an interview with a
private individual. With your own material in a working directory:

```bash
pip install faster-whisper numpy

# 1. baseline transcript
python scoring/transcribe.py your-audio.wav claude_transcript

# 2. place alongside it: manual.txt, sinribe.md, vibe.txt
#    (docx inputs converted to text, one paragraph per line)

# 3. score
SINRIBE_BENCH_DIR=$PWD python scoring/analyze.py   # WER, error tiers, runs, timeline
SINRIBE_BENCH_DIR=$PWD python scoring/timing.py    # timestamp accuracy
SINRIBE_BENCH_DIR=$PWD python scoring/final.py     # diarization, agreement, segmentation
```
