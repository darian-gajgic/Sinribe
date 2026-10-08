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
`vorlesung.srt`, `vorlesung.vtt`, and a `vorlesung - Summary.html` briefing page:

```markdown
# vorlesung

**Duration** 2:14:33 · **Speakers** 3 · **Language** de (0.98)
**Model** large-v3 · CUDA · float16 · **Diarization** speaker-diarization-community-1 · **Quality** 20× target
**Source** `~/Recordings/vorlesung.m4a`
**Transcribed** 2026-07-26 18:03 · 14m 21s (21.0× realtime transcribing)

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

At the end, unless you turn it off, comes a **Worth a listen** section: the passages the
recogniser was least sure of, with timestamps and the doubtful words in bold.

```markdown
## Worth a listen

- **[00:14:52]** Person 2: schon ein elementares Element. **Bei einer** Nasenatmung zum Beispiel
- **[00:31:07]** Person 1: wenn das in eurem **Ruf steht.**
```

Fifty-seven percent of the words it flags are genuinely wrong, so on a difficult recording this is
where an hour of checking gets you the most back — considerably more than any decode setting will.
Every flagged passage is listed, in time order, so the section doubles as a checklist to work
through against the audio; a hard 65-minute interview produces a few hundred. Set
`review_max_spans` in the config if you would rather have only the *n* least confident.

## Summary and HTML presentation

Tick **Summary + HTML presentation** and the job does a second piece of work after the
transcript: it writes a brief you can learn from instead of listening to the whole recording,
saved next to the transcript as `<name> - Summary.html` and `<name> - Summary.md`.

The page is ordered for someone who wants the substance fast:

| Part | What it gives you |
|---|---|
| **The 1-minute version** | the whole idea in one paragraph, 4 to 6 key takeaways, why it matters, and the numbers worth remembering |
| **What has changed since it was published** | only with the web check: every checked claim marked *outdated*, *incorrect*, *disputed*, *still open*, *still accurate* or *unverified*, what is true now, and since when |
| **Key terms** | the vocabulary a newcomer needs, before the detail that uses it |
| **The full summary** | one section per topic in the order the recording covers it: the point in one sentence, two to four paragraphs that explain it, the key points, and optional boxes for examples, evidence, forecasts, caveats, how-tos and definitions |
| **Worth watching in the original** | two to four moments where the delivery or a demo matters |
| **Test yourself** | five to eight questions with the answers folded away |

The full summary is budgeted to stay well short of listening: about 15 % of the words spoken,
between 400 and 3,000 words, so a ten-minute video gets about a page and a two-hour interview
about a quarter of an hour of reading. The top of the page says how long it takes to read against
how long the recording is, and how current it is: when the video was published, and either the web check's verdict counts or a plain
"not checked against the web". Every timestamp links into the video at that moment when the
source is YouTube, so any claim can be heard in the speaker's own words.

The page is one self-contained file. No JavaScript, no CDN, no fonts fetched: it opens from a
`file://` path, prints cleanly, and will still render on a machine with no network.

En and em dashes are stripped from everything the model writes: between digits they become a
hyphen, spaced they become a comma. They are the clearest tell that a page was machine-written,
and asking a model not to use them only works about half the time.

### Who writes it

| Provider | Needs | What you get |
|---|---|---|
| **Claude Code** (default) | nothing: the `claude` CLI and the login you already have | The whole recording in one context. Costs subscription usage, no API key and no separate bill |
| Claude API | `ANTHROPIC_API_KEY`, or the key in `~/.config/sinribe/anthropic_key`, or `ant auth login` | The same models over the direct API, which enforces the output shape in the decoder rather than asking for it. Billed per token |
| Local Ollama | `gemma3:4b` already running on `:11435` | Offline and free. Reads the recording in chunks and writes fewer, thinner sections |

`auto` tries them in that order, so on a machine where you already use Claude Code the checkbox
just works. A stock Ollama install listens on port 11434, not 11435; if yours does, set
`"llm_url": "http://127.0.0.1:11434"` in `~/.config/sinribe/config.json`.
Whichever ran is named in the page's footer along with what it cost, so you can always
tell what you are reading.

**Claude Code needs no key because it is the same thing you would do by hand.** It runs
`claude -p` and signs in with the login on this machine. Each job is one resumed session: the
transcript is sent once and every later pass continues that session, so the transcript is read
from cache (measured: 2 input tokens against 11,677 read from cache on a resumed turn) and each
section is written with the earlier sections already in view, which is what keeps section 9 from
restating section 2. The CLI runs `--restricted` from a scratch directory, so the summariser has
no shell, no code execution and no project `CLAUDE.md` to inherit.

The default local model, `gemma3:4b`, is the honest minimum rather than a recommendation. Measured
on a real interview it writes a complete page for 25 minutes of audio in about two minutes, but it
is visibly weaker than the Claude path and it fails in ways a bigger model does not: on a
100-second clip it produced a plan with no usable sections at all. If you mean to use the offline
path regularly, pull something larger, which this machine's 16 GB of VRAM has room for:

```bash
ollama pull qwen3:14b
# then set "llm_model": "qwen3:14b" in ~/.config/sinribe/config.json
```

Nothing is guessed silently. The provider is probed *before* the audio is decoded, using
`claude auth status` or a local credential read rather than a billed request, so a missing login
costs you a dialog rather than the whole job. A summary that fails after the transcript is written
says so in the window instead of just not appearing.

### Web research

**Verify with web research** checks the recording against what is true *today*. That matters
most for AI: a two-month-old video about models, prices or company plans can already be wrong.

1. While planning the brief, the model lists up to 16 specific, checkable claims from the
   recording, time-sensitive ones first (model versions, benchmarks, prices, release dates,
   company plans, predictions with a date), each with its timestamp.
2. A separate research call searches the web and opens pages to find what is true now. It is told
   the video's publication date, prefers primary sources and sources newer than the video, and
   may not call a claim outdated on an undated source alone.
3. A tool-free call turns the report into one verdict per claim, plus the notable developments
   since the recording.
4. The brief is then written with those verdicts in hand: an outdated claim is stated as said
   *and* corrected where it comes up, flagged at the top of its section, listed in "What has
   changed", and repeated as a heads-up inside the 1-minute version so a reader who stops there is
   still warned.

Links are held to what the search tools actually returned, not to what the model wrote. Every
verdict other than *unverified* needs at least one such source or it is downgraded to
*unverified*, links inside the prose are unlinked if they are not on the list, and the Sources
section is filtered the same way. If the web check fails (a timeout, the CLI dying), the brief is
still written from the transcript and says plainly at the top that the check did not run.

It needs one of the two Claude providers; the local model has no network access, and asking for
research with it is refused before the job starts rather than quietly skipped.

```bash
sinribe "https://youtube.com/watch?v=..." --research           # implies --summary
sinribe lecture.m4a --summary                                  # default: Claude Code, no key
sinribe lecture.m4a --summary --summary-provider ollama        # fully offline
```

### What a summary costs

Measured on 8 October 2026 with the current pipeline, a 10m45s video with the web check on:
**5 minutes 24 seconds and $3.25 of usage** for 3 sections, 14 claims checked and a three-minute
read. The figure below is from before the rework, on Opus 5; expect the long case to cost less
now that sections have a word budget, but it has not been re-measured.

Measured on the two-hour interview this feature was built against, with research on:
**18 minutes and $17.97 of usage**, for a 104 kB page with 12 sections, 72 blocks, 36 quotes and
17 verified links. That is fifteen Opus calls, and each section turn re-reads the conversation so
far, so the cost grows with the number of sections. The page footer and the log both report the
figure for the run you actually did.

If that is more than a recording is worth to you, the levers in order of effect:

| Lever | How | Effect |
|---|---|---|
| Skip the research pass | leave **Verify with web research** unticked | Removes the web turns and the source-gathering |
| Fewer sections | `"summary_max_sections": 8` in the config | Roughly linear: 8 sections cost about two thirds of 12 |
| Fewer claims checked | `"summary_max_claims": 8` in the config | Shortens the research pass |
| A cheaper model | `"summary_model": "claude-sonnet-5-5"` | Much cheaper per token, visibly less incisive |
| Nothing at all | `--summary-provider ollama` | Free and offline, and much thinner |

A short recording is cheap either way: the section count scales with duration, so a ten-minute
clip gets three sections rather than twelve.

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

Sinribe is built and tested on Ubuntu with an NVIDIA GPU. Other Linux distributions should work.
Without an NVIDIA GPU it runs on the CPU, which works but is many times slower. Plan for about
13 GB of disk: two virtualenvs (8.4 GB) and the speech models (4.5 GB).

You need `git`, `ffmpeg` and [uv](https://docs.astral.sh/uv/), which fetches Python 3.12 by itself:

```bash
sudo apt install git ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/darian-gajgic/Sinribe.git
cd Sinribe
```

pyannote's models are gated, so there is a one-time online step:

1. Create a HuggingFace account: <https://huggingface.co/join>
2. Accept the licence at <https://huggingface.co/pyannote/segmentation-3.0>
3. Accept the licence at <https://huggingface.co/pyannote/speaker-diarization-community-1>
4. Make a **Read** token at <https://huggingface.co/settings/tokens>
5. `./install.sh --hf-token hf_xxxxxxxx`

The installer builds both virtualenvs, downloads the diarization models and `large-v3`, converts
the German turbo model that the default quality setting votes with, and adds a `sinribe` command
and a desktop icon. Start it with `sinribe`, or `~/.local/bin/sinribe` if that folder is not on
your `PATH` yet.

After that everything runs with `HF_HUB_OFFLINE=1`. The token is stored in
`~/.config/sinribe/hf_token` (chmod 600) and is never written into this repo.

The summary is optional. It uses [Claude Code](https://claude.com/claude-code) if it is installed
and logged in, an Anthropic API key, or a local model through Ollama; see
[Who writes it](#who-writes-it).

If the window does not open and the error mentions `xcb`, install the one Qt library Ubuntu
leaves out: `sudo apt install libxcb-cursor0`.

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

| | Sinribe | Stock large-v3 | Vibe |
|---|---|---|---|
| Word error rate | **18.84%** | 20.17% | 20.51% |
| Meaning-changing errors | **7.46%** | 8.10% | 8.20% |
| Median timestamp error | **1.3 s** | 2.0 s | 14.0 s |
| Speaker attribution | **99.60%** | — | — |

The stock large-v3 baseline is plain `faster-whisper large-v3` with no diarization, run for this
benchmark by Claude Code (an AI coding agent). Sinribe uses the same base model, so the gap measures
its pipeline, not a better model.

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

Re-export does not touch the summary page. Its quotes carry whatever speaker names the transcript
had when it was written, and regenerating it means another run of the model rather than a rewrite
of a file, so renaming speakers afterwards leaves the old names in the summary. Rename first if
you care, or re-run with the summary ticked.

## Options

| Option | Default | Notes |
|---|---|---|
| Quality | 10× realtime ★ | The slider *is* the speed, 84× down to 1×. Each position shows its measured error rate |
| Model | `auto` | The quality setting picks one. Override with `large-v3`, a German model, `medium.en`, `small` |
| Names & terms | empty | Proper nouns to expect. Helps with names the model nearly gets; see below |
| Language | Auto-detect | Or pin it, which is slightly faster and safer on quiet openings |
| Speakers | Auto-detect | Or set exactly N, or a min–max range |
| Flag uncertain passages | on | Appends the spots worth checking by ear, with timestamps. 57 % of what it flags is genuinely wrong |
| Sidecar `.json` | on | Needed for instant re-export |
| `.srt` / `.vtt` | on | Speaker-prefixed cues |
| LLM chapters + summary | off | Uses `gemma3:4b` on your local Ollama (`:11435`); still offline |
| Summary + HTML presentation | off | Writes a briefing on the recording and renders it as a standalone web page. See below |
| Verify with web research | off | Lets the summary check claims against the web and collect real sources. Needs a Claude provider |
| Written by | `auto` | `auto`, `claude-code` (no key needed), `claude` (the API) or `ollama` (local) |
| `yt_cookies_from_browser` | `""` | Config-file only. Set to `"firefox"` etc. for gated episodes |
| `keep_downloads` | `true` | Set false to expire cached downloads after `download_cache_days` |

Settings live in `~/.config/sinribe/config.json`.

### What the quality slider is, and what it is not

Its value is the target realtime factor, so "20×" means an hour of audio in three minutes. Each
position carries the error rate it actually scored on a hard German interview, and the estimated
time for the file you have loaded — so it reads *"20.4× realtime · 21.3% errors · ≈ 3m 13s for
this file"* rather than *"high quality"*.

| Slider | Passes | Speed | Errors | |
|---|---|---|---|---|
| 84× | 1 | 85× | 23.1 % | turbo German model, batched |
| 50× | 1 | 51× | 21.6 % | `large-v3`, batched |
| 20× | 1 | 20× | 21.3 % | `large-v3`, one pass in order |
| 14× | 2 | 15× | 20.8 % | + a batched pass, voted |
| **10× ★** | 3 | 11× | **20.4 %** | + a second *model*, voted |
| 5× | 5 | — | not scored | + a pass on noise-reduced audio |
| 2× | 7 | — | not scored | + a cleaned pass and a wider beam |
| 1× | 9 | — | not scored | + a very wide beam — the beam is measured to buy nothing |

**Measured through three passes; uncalibrated below that.** The figures the slow rungs used to
carry were withdrawn on 2026-08-07: they were measured while a bug fed those rungs' conditioned
passes the *unfiltered* audio, so passes that were supposed to hear the recording differently
returned the first pass's words verbatim and voted twice. The bug is fixed and those rungs now
run what they advertise — but what they advertise has never been scored, so the slider shows them
as targets rather than as measurements. The ★ rung is unaffected: its three passes were always
genuinely different, and 20.4 % is its own measured number.

### How the slow half works, and why the fast half doesn't just do it

Below 20× the audio is transcribed **several times with different settings, and every word is
decided by majority vote** (`sinribe/rover.py`). Passes that chunk the audio differently mishear
*different* words, so where two agree against a third the majority is usually right. That takes
21.3 % down to 20.4 %, and no further.

The single biggest step is adding a **different model** (20.8 % → 20.4 %), not re-running the same
one with different chunking. Two models trained differently make different mistakes; one model
re-segmented mostly makes the same ones.

What does *not* work is simply searching harder. A beam of 20 costs 5.6× the compute and returns
21.6 % — worse than the default. Beam search explores one model's own hypotheses, so it only
rescues an answer the model already had; here the model's probabilities are themselves wrong about
whether it heard *das* or *es*, and searching a wrong distribution harder returns the same wrong
answer at five times the price.

Whether the passes past the third are worth their time is currently unknown — see the note under
the table. The ★ sits at the last point on the curve with a measured number behind it.

Timings are **transcription only**: speaker diarization is excluded, because on a re-run it comes
from cache and including it would make the same setting look like a different speed depending on
what was already on disk. A first pass over a new file adds roughly two minutes for that stage.
They came off one machine on one day — the app replaces them with what your own hardware actually
does, after your first run at each setting that decodes every one of its passes. A run that
resumes from cache reports what it reused instead of inventing a speed from it.

### Names & terms — worth trying, not a cure

Whisper mangles proper nouns it has no reason to predict. In one interview *Fujitsu* came back as
"fiuzi", "jitze" and "future service" — never once correctly — while *VR-Bank* became "Feuerbank".
Listing the expected names here biases the recogniser's vocabulary:

```
Fujitsu, Siemens, VR-Bank, Ramada, Dr. Mühlbauer
```

Measured on that interview it is a weak lever. A name the recogniser was already nearly getting
("Dieter", right once out of three) came good; *Fujitsu* and *VR-Bank* did not move at all, and
overall WER rose slightly, 21.3 % → 22.3 %. A vocabulary hint is no match for a word the model
confidently mis-hears. Try it, then check whether it earned its place — `sinribe-eval` will say.

### The German models are optional, and not automatically better

`large-v3-german` and `large-v3-turbo-german` are fine-tuned on German alone, where vanilla
`large-v3` is multilingual. That sounds like a straight upgrade and is not one: on a hard
far-field Bavarian interview the German fine-tune scored **24.8 % WER against large-v3's 21.3 %**,
with more substitutions *and* more invented words. Its published numbers come from read speech,
and the narrower training appears to cost large-v3's robustness on spontaneous, distant, accented
audio.

They are worth trying on clean, close-miked German — and `sinribe-eval` will tell you, which is
the point. Install them with:

```bash
tools/convert_german_models.sh          # both, ~4.7 GB
tools/convert_german_models.sh turbo    # just the fast one, ~1.6 GB
```

The upstream models ship in transformers format, so this converts them to CTranslate2. It runs in
a throwaway virtualenv on purpose — see `HANDOFF.md` for why nothing may install torch into
`.venv`.

## Measuring accuracy

Transcription settings are easy to argue about and easy to measure, so Sinribe ships the scorer:

```bash
./sinribe-eval reference.docx transcript.md
```

It aligns the two word by word and reports WER, per-speaker accuracy, how the error rate moves
over the recording, speaker-attribution accuracy, and the most frequent confusions. The reference
can be `.docx`, `.odt`, `.md`, `.txt`, `.srt` or `.vtt`; timestamps that restart part-way through
(a document stitched from two recordings) are handled, and `(unverständlich)` markers are dropped
rather than counted against the recogniser. German spelling variants — `softskills`/`soft skills`,
`gernhabt`/`gern habt` — score as correct, because they are.

To compare settings instead of grading one run:

```bash
./sinribe-eval --sweep variants.json audio.wav reference.docx -o results.json
```

Each variant re-decodes the audio and is scored; the decoded audio and the diarization are shared
between variants, so only the recognition pass is repeated.

And to answer "is more tuning worth it?" rather than "which of these is best?":

```bash
./sinribe-eval --ceiling ~/.cache/sinribe/sweep reference.docx
```

That scores every transcript it finds and reports what a *perfect* choice between them would
score. The gap to the best single configuration is the entire prize available to any cleverer
scheme; the words none of them get right are the floor. On the benchmark interview the floor is
8.7 % of all words — fast, broad-dialect, far-field speech that no setting reaches — which is how
two-model consensus decoding got cancelled before it was built.

Downloaded podcasts are large and are kept indefinitely by default. To reclaim space, either
delete `~/.cache/sinribe/downloads/` or set `"keep_downloads": false`.

## Against Vibe

Vibe is another desktop transcription app, so it is the honest thing to measure against rather
than against a number from a paper. Both were given the same file, and both outputs were scored
against the same human transcript with `./sinribe-eval`.

The file was chosen to be punishing: a 1 h 05 m coach interview in **broad Bavarian**, badly
recorded — two speakers, a real room, no headset, no studio. Neither error rate below is what
either tool does on clean audio, and neither is meant to be. Strong regional accent over poor
audio is the case that actually separates recognisers; clean speech is the case where everything
scores well and nothing is learned. The reference is a human transcript of the **whole** recording
— not a sampled stretch of it — 8980 words after fillers and `(unverständlich)` markers are
dropped from both sides. It arrives in two parts whose timestamps restart at zero, so part two's
`[00:10:52]` is audio minute 36:51; the scorer detects that and aligns on word sequence rather
than on the reference clock.

Three Sinribe runs are in the table, because the useful question is not only whether it beats Vibe
but which setting you should actually leave the slider on.

| metric | vibe | **10× ★, 3 passes** | 1×, 9 passes | 2 passes (Win) | manual |
|---|---|---|---|---|---|
| words | 9503 | 9316 | 9361 | 9269 | 8980 |
| word match | 86.4 % | **86.7 %** | 86.7 % | 85.2 % | ref |
| substitutions | 830 | 766 | 783 | **759** | ref |
| deletions | **390** | 432 | 411 | 571 | ref |
| insertions | 913 | **768** | 792 | 860 | ref |
| **WER** | 23.8 % | **21.9 %** | 22.1 % | 24.4 % | ref |
| 5-min buckets won (13) | 2 | 3 | **7** | 1 | — |
| segments | 929 | 160 | 160 | 169 | 104 |
| named entities (of 15) | 15 | 15 | 15 | 15 | 15 |
| speaker accuracy | none | **97.3 %** | 97.3 % | 95.9 % | ref |
| timestamp drift (median) | 7.0 s | **2.5 s** | 2.5 s | 3.0 s | ref |
| — within 3 s | 4/20 | 11/20 | 11/20 | 11/20 | ref |
| duplicate words /1k | **1.4** | 4.1 | 1.9 | 4.7 | 0.9 |
| repeat runs ≥ 3 | **0** | 5 | 0 | 5 | 1 |
| `" -"` spacing bug | **0** | 28 | 33 | 28 | 0 |
| run-on unpunctuated blocks | 18 | 3 | **2** | 5 | 1 |
| runtime | n/a | **2 m 14 s** (29.5×) | 31 m 00 s (2.1×) | 9 m 09 s (7.5×) | — |

**The recommended setting is also the best one.** The ★ rung — three voted passes — scores 21.9 %
in 2 m 14 s. The 1× rung spends 31 minutes, fourteen times as long, to land 0.2 pp *behind* it.
That is the ladder's own conclusion arrived at from the other direction, on a recording it was
not tuned on.

**Where Vibe is genuinely better.** It deletes fewer words (390 against 432), and its text is
cleaner: 1.4 duplicated words per thousand against 4.1, no repeated runs of three or more against
five, and none of the `" -"` spacing artifacts Sinribe emits 28 times in this transcript. That
last one is a rendering bug on this side, not a recognition result, and it is not fixed yet.

**Where the gap is not close.** Vibe returned no speaker labels at all, so a two-person interview
arrives as one undifferentiated wall of text; Sinribe places 97.3 % of it correctly. Its
timestamps drift a median 7.0 s against 2.5, with 4 of 20 checked cues landing within three
seconds against 11. And it emits 929 segments where the human transcriber made 104 — the
transcript is shredded into fragments, with 18 run-on unpunctuated blocks against 2 — so what you
have to read afterwards is worse than the word error rate on its own suggests. Both tools found
all 15 named entities.

**How to read the bucket row.** The recording is cut into thirteen five-minute windows and each
window goes to whichever of the four transcripts scored best in it, so the four columns sum to 13.
The 7 against the 1× column therefore means "best of four in seven windows", not "beat Vibe seven
times" — the Sinribe runs are mostly taking windows off each other.

Three caveats worth stating. The 2-pass column was run on Windows from a different encode of the
same interview — a 1 h 08 m `.ogg` rather than the `.wav` the others used — so its deletion count
especially is not strictly comparable to the rest of the row. The runtimes are transcription only,
with diarization served from cache; a first run over a new file adds roughly two minutes for that
stage. And this is one recording: hard, German, real, but one. It reports what happened on this
interview, not what will happen on yours — which is what the scorer is shipped for.

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
  summary/               the briefing: providers (Claude API / Ollama), passes, output schema
  textfmt.py             sentence splitting, timestamp formatting
  workers/asr_worker.py  faster-whisper subprocess   (.venv)
  workers/diar_worker.py pyannote subprocess         (.venv-diar)
  render/                markdown, subtitles, sidecar, the summary web page
  ui/                    PySide6 window, worker thread, speaker panel
```

## Credit where it is due

Reuses hard-won pieces from earlier tools on this machine: the CUDA loader shim and the EN/DE
sentence splitter from `local-wisprflow`, the colour palette and config handling from `Sinlate`,
and the "one model per subprocess, VRAM is freed by process exit" lesson from `Nexus`.
