"""The quality/speed ladder behind the accuracy slider.

The slider's value IS the target realtime factor: 20× means "an hour of audio in three minutes",
1× means "take as long as the recording lasts". Each rung names what it runs and carries the error
rate that actually scored, so the slider says "10.7× realtime · 20.4% errors" instead of "high
quality" — a claim the user can check rather than take on trust.

The slow half of the ladder runs the audio through SEVERAL decodes and votes on every word
(`rover.py`). That is the only mechanism found that converts time into accuracy: 21.3 % for one
pass, 20.4 % for three. Widening the beam instead does nothing at all.

Voting stops paying after about the third pass. Through three passes that is measured and holds.
Past three it is currently UNMEASURED, and the numbers that used to sit here (20.5 % at five,
20.6 % at seven and nine) have been withdrawn — see the note on the slow rungs below. They were
produced by a run in which the denoise and clean passes silently decoded UNFILTERED audio, so
they returned output byte-identical to the pivot: at five passes the pivot cast two of the five
votes and at seven or nine it cast three. "The extra passes duplicate a model that has already
voted" was therefore not a finding about voting, it was a description of a bug (fixed 2026-08-07
in `pipeline._decode`, which now hands each pass the audio its own settings ask for).

What survives that correction: the three-pass rung, whose passes were always genuinely distinct,
and the shape of the curve up to it — 21.3 % for one pass, 20.8 % for two, 20.4 % for three.
Whether five real passes beat three is an open question again, and answering it means re-running
`python -m sinribe.eval --sweep` on the benchmark interview.

Everything here is calibrated by `python -m sinribe.eval --sweep` against a human transcript of a
deliberately hard recording. Two rules the calibration learned the hard way, both in HANDOFF.md:
run nothing else on the GPU, and interleave the variants rather than running them in blocks. Error
rates are reproducible to the byte; wall clock is not, which is why `config.observed_rtf` replaces
these timings with whatever the user's own machine turns out to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The decode contract: every knob the ASR stage understands, with the values that reproduce
# Sinribe's behaviour before the accuracy work. Rungs override what they change.
BASE_DECODE: dict = {
    "asr_mode": "accurate",          # accurate = sequential decode; fast = batched
    "beam_size": 5,
    "patience": 1.0,
    # OFF, and measured that way. Feeding whisper's own previous output back as context is the
    # textbook fix for the function-word errors that dominate here, and on the benchmark
    # interview it made things clearly worse: WER 21.3 % -> 24.4 %, deletions 395 -> 749, and
    # speaker attribution fell from 98.9 % to 94.2 % because the lost words took the timings with
    # them. Adding repetition guards on top did not rescue it. Whisper's context window carries
    # its own errors forward, and on hard spontaneous speech there are enough of them for that to
    # be a net loss. Do not switch this back on without re-running the sweep.
    "condition_on_previous_text": False,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "prompt_reset_on_temperature": 0.5,
    # When a window decodes badly (low average logprob, or output so repetitive it looks like a
    # loop), whisper re-decodes it at rising temperature. That trades a beam search for sampling,
    # which is the right call on a window that is genuinely broken and the wrong one on a window
    # that is merely hard — and hard is this recording's normal state.
    "temperature_fallback": True,
    "vad_min_silence_ms": 500,
    "vad_speech_pad_ms": 400,
    "audio_filter": "",              # "" | "clean" | "denoise"
}


@dataclass(frozen=True)
class Rung:
    """One position on the slider."""
    target_rtf: int
    model: str                       # used when the model dropdown is on "auto"
    note: str
    decode: dict = field(default_factory=dict)
    # Extra decodes to run and vote on, beyond `decode` itself. Each entry is a set of overrides
    # that makes the pass hear the audio differently — different silence boundaries, different
    # chunking, conditioned audio. Voting only pays when the passes disagree, so these vary
    # segmentation rather than search width.
    extra_passes: tuple[dict, ...] = ()
    measured_rtf: float | None = None
    measured_wer: float | None = None
    # Exactly one rung carries this. It is where the accuracy curve flattens — past it you are
    # paying real minutes for fractions of a percent — so it is what a user who does not want to
    # think about any of this should get.
    recommended: bool = False

    def settings(self) -> dict:
        return {**BASE_DECODE, **self.decode}

    def label(self) -> str:
        """What the slider prints. Falls back to the plain target until calibrated.

        One decimal on the WER, because the rungs sit fractions of a point apart and rounding to
        whole percents would print several of them as the same number.
        """
        rtf = self.measured_rtf or self.target_rtf
        if self.measured_wer is None:
            return f"{rtf:g}× realtime"
        text = f"{rtf:g}× realtime · {self.measured_wer:.1%} errors"
        return f"{text}  ★ recommended" if self.recommended else text

    def duration_for(self, seconds: float, observed_rtf: float | None = None) -> float:
        """How long this rung takes on `seconds` of audio.

        `observed_rtf` — what this machine actually did last time — wins over the shipped figure.
        It has to: the benchmark numbers came off one machine on one day, and the same settings
        there measured 50× and 20× depending on what else was resident on the GPU.
        """
        rtf = observed_rtf or self.measured_rtf or self.target_rtf
        return seconds / rtf if rtf else 0.0


# Ordered fastest first. Every figure here was measured by `--sweep` on the benchmark interview
# (65 min of far-field Bavarian German), each setting timed at least twice, interleaved with the
# others so that GPU drift could not masquerade as a difference. Sequential landed on 193 s and
# 194 s, batched on 77 s and 78 s.
#
# The curve, as produced BY THIS CODE on the benchmark interview:
#
#   passes   1       2       3       5
#   speed    20.4×   14.6×   10.7×   5.4×
#   WER      21.3%   20.8%   20.4%   20.5%
#
# So spending more time DOES buy accuracy — through voting, not through searching harder. The
# passes chunk the audio differently and mishear different words; a majority vote keeps what two
# of them agree on. Widening the beam instead does nothing: beam 20 costs 5.6× and returns
# 21.6 %, because beam search only rescues an answer the model already had, and here the model's
# probabilities are themselves wrong.
#
# Above three passes the ladder is UNCALIBRATED. The 5x, 2x and 1x rungs carried measured figures
# until 2026-08-07, when the passes they added turned out not to be the passes they claimed: the
# audio filter was read from the job config rather than the pass's, so "denoise" and "clean"
# decoded the untouched WAV and returned the pivot's own words back to the vote. Those rungs are
# now running something that has never been scored, and a number that has not survived the real
# implementation does not get printed as if it had — that mistake has already been made twice in
# this file's history, so the numbers come off rather than being quietly kept.
#
# Timings assume the speaker diarization is already cached, which is what a re-run costs; a first
# pass over a new file adds roughly two minutes for that stage. They are also from one machine on
# one day — `config.observed_rtf` overrides them with what the user's own hardware actually does.

# The passes the voting rungs draw on. Each must hear the audio differently — voting can only fix
# a word the passes disagree about, so a pass that agrees with the pivot everywhere is pure cost.
#
# A greedy search over twelve real decodes picked these, in this order, and the ordering is the
# lesson: the biggest single gain came from adding a DIFFERENT MODEL (21.3 % -> 20.8 % -> 20.4 %),
# not from re-running the same one with different chunking. Two models that were each trained
# differently mishear different words; one model re-segmented mostly mishears the same ones.
# A pass may therefore override "model" as well as any decode setting.
P_BATCHED = {"asr_mode": "fast"}
P_TURBO = {"model": "large-v3-turbo-german"}
P_TURBO_BATCHED = {"model": "large-v3-turbo-german", "asr_mode": "fast"}
P_DENOISE = {"audio_filter": "denoise"}
P_CLEAN = {"audio_filter": "clean"}
P_WIDE_VAD = {"vad_speech_pad_ms": 700, "vad_min_silence_ms": 300}
P_BEAM8 = {"beam_size": 8, "patience": 2.0}
P_BEAM20 = {"beam_size": 20, "patience": 2.0}

RUNGS: list[Rung] = [
    Rung(target_rtf=84, model="large-v3-turbo-german",
         note="Fastest. A smaller German model, batched. Gets about two words in a hundred "
              "wrong that the careful setting gets right — good for finding your way around a "
              "long recording, not for quoting from it.",
         decode={"asr_mode": "fast", "beam_size": 5},
         measured_rtf=85.2, measured_wer=0.231),
    Rung(target_rtf=50, model="large-v3",
         note="Batched: the audio is split at silences and the pieces decoded in parallel, "
              "which costs about one word in a hundred at the seams. Speaker labelling is "
              "unaffected.",
         decode={"asr_mode": "fast", "beam_size": 5},
         measured_rtf=50.9, measured_wer=0.216),
    Rung(target_rtf=20, model="large-v3",
         note="One pass, decoded in order. The best a single decode gets — fifteen "
              "configurations were measured against a human transcript and none beat it.",
         decode={"beam_size": 5},
         measured_rtf=20.4, measured_wer=0.213),
    Rung(target_rtf=14, model="large-v3",
         note="Two passes, voted word by word. The second decodes the audio in parallel chunks, "
              "so it breaks the speech in different places and mishears different words.",
         decode={"beam_size": 5},
         extra_passes=(P_BATCHED,),
         measured_rtf=14.6, measured_wer=0.208),
    Rung(target_rtf=10, model="large-v3",
         note="Three passes, voted — and the third is a DIFFERENT MODEL. That is where most of "
              "the gain comes from: two models trained differently mishear different words, so "
              "the majority is right more often than either alone.",
         decode={"beam_size": 5},
         extra_passes=(P_BATCHED, P_TURBO),
         measured_rtf=10.7, measured_wer=0.204,
         recommended=True),
    # The three rungs below add noise-conditioned passes. Their old scores were measured while
    # those passes were silently decoding unfiltered audio, so they described a different ladder
    # than the one that now runs; the figures are withdrawn until a sweep re-earns them. Until
    # then the slider shows these as targets, not measurements, which is the honest state.
    Rung(target_rtf=5, model="large-v3",
         note="Five passes across two models, one of them on noise-reduced audio. NOT YET "
              "SCORED: the previous measurement was taken while the noise reduction was not "
              "actually being applied, so it described five passes of which one was a copy. "
              "Three passes is the setting with a number behind it.",
         decode={"beam_size": 5},
         extra_passes=(P_BATCHED, P_TURBO, P_TURBO_BATCHED, P_DENOISE)),
    Rung(target_rtf=2, model="large-v3",
         note="Seven passes, two of them on conditioned audio. NOT YET SCORED for the same "
              "reason as the setting above, and slow enough that it should not be picked on "
              "faith: expect roughly four times the recommended setting's wait.",
         decode={"beam_size": 5},
         extra_passes=(P_BATCHED, P_TURBO, P_TURBO_BATCHED, P_DENOISE, P_CLEAN, P_BEAM8)),
    Rung(target_rtf=1, model="large-v3",
         note="Nine passes including a very wide beam search. Slowest available, NOT YET "
              "SCORED, and the beam-20 pass alone takes longer than the entire recommended "
              "setting — while beam width is the one thing measured to buy nothing at all. "
              "Kept so the ladder ends somewhere; do not use it.",
         decode={"beam_size": 5},
         extra_passes=(P_BATCHED, P_TURBO, P_TURBO_BATCHED, P_DENOISE, P_CLEAN, P_BEAM8,
                       P_BEAM20, P_WIDE_VAD)),
]

MIN_TARGET = min(r.target_rtf for r in RUNGS)
MAX_TARGET = max(r.target_rtf for r in RUNGS)


def recommended() -> Rung:
    """The rung to start a new user on, and the one the slider marks."""
    return next((r for r in RUNGS if r.recommended), RUNGS[-1])


def observed_rtf(cfg: dict, rung: Rung) -> float | None:
    """What this machine actually achieved on this rung, if it has run one."""
    try:
        return float((cfg.get("observed_rtf") or {})[str(rung.target_rtf)])
    except (KeyError, TypeError, ValueError):
        return None


def trustworthy_rtf(result: dict) -> float:
    """The transcription speed a finished job actually demonstrated — 0.0 if it demonstrated none.

    A rung's speed is the cost of running ALL of its passes. A job that decoded two of three and
    read the third off a checkpoint ran at no rung's speed, and neither did one that reused the
    single pass it was asked for. Reporting its wall clock as a realtime factor is how a run at
    the 10.7x rung came to print 29.5x — which reads, correctly, as the setting being ignored.

    Diarization being cached does NOT disqualify a run: the figures in this file were measured
    that way on purpose, so transcription-only is the comparison that holds.
    """
    ran = int(result.get("asr_passes_ran") or 0)
    total = int(result.get("asr_passes_total") or 0)
    rtf = float(result.get("asr_realtime_factor") or 0.0)
    return rtf if total and ran == total and rtf > 0 else 0.0


def speed_note(result: dict) -> str:
    """The speed clause for a log line or a transcript header, or "" when there is nothing to say.

    Every surface that prints a speed goes through here, so the log, the window and the transcript
    cannot end up telling the user three different stories about the same run.
    """
    rtf = trustworthy_rtf(result)
    if rtf:
        return f" ({rtf:.1f}× realtime transcribing)"
    ran = int(result.get("asr_passes_ran") or 0)
    total = int(result.get("asr_passes_total") or 0)
    if total and ran == 0:
        return " (transcription reused from cache — this is not a transcription speed)"
    if total and ran < total:
        return f" ({ran} of {total} passes decoded, the rest reused from cache)"
    if result.get("reused_stages"):
        return " (partly reused from cache — this is not a transcription speed)"
    return ""


def record_rtf(cfg: dict, speed_target: float, realtime_factor: float) -> None:
    """Fold a finished job's speed into the running estimate for its rung.

    `realtime_factor` must be the transcription-only figure from `trustworthy_rtf` — the same
    quantity `measured_rtf` holds. Feeding it end-to-end wall clock mixes two different
    measurements under one key, and feeding it a cache-served run teaches the slider to promise a
    speed nothing can deliver. A zero is ignored, which is how a run with nothing honest to
    report says so.

    Averaged with what was there rather than replacing it, so one job that happened to run while
    Ollama was loading a model does not become the number every future estimate is built on.
    """
    if not realtime_factor or realtime_factor <= 0:
        return
    rung = for_target(speed_target)
    seen = dict(cfg.get("observed_rtf") or {})
    key = str(rung.target_rtf)
    previous = seen.get(key)
    try:
        seen[key] = round((float(previous) + realtime_factor) / 2, 2) if previous \
            else round(realtime_factor, 2)
    except (TypeError, ValueError):
        seen[key] = round(realtime_factor, 2)
    cfg["observed_rtf"] = seen


def for_target(target_rtf: float) -> Rung:
    """The rung for a slider position.

    Between two rungs the SLOWER one wins: the slider promises a speed ceiling, and quietly
    running faster than asked by dropping accuracy is the wrong way to miss.
    """
    target = max(MIN_TARGET, min(MAX_TARGET, float(target_rtf)))
    eligible = [r for r in RUNGS if r.target_rtf <= target]
    return max(eligible, key=lambda r: r.target_rtf) if eligible else RUNGS[-1]


def decode_settings(cfg: dict) -> dict:
    """Resolve one job's decode settings: rung defaults, then any explicit overrides.

    `decode_overrides` is how the sweep drives a single knob without inventing a config key for
    every experiment; ordinary runs never set it.
    """
    rung = for_target(cfg.get("speed_target", 6))
    settings = rung.settings()
    overrides = cfg.get("decode_overrides") or {}
    settings.update({k: v for k, v in overrides.items() if k in BASE_DECODE})
    return settings


def model_for(cfg: dict) -> str:
    """The model this job should use: the dropdown's choice, or the rung's when it says auto."""
    chosen = str(cfg.get("asr_model") or "auto")
    if chosen != "auto":
        return chosen
    override = (cfg.get("decode_overrides") or {}).get("model")
    return str(override or for_target(cfg.get("speed_target", 6)).model)
