"""Settings persistence for Sinribe.

Whitelist-filtered load + atomic write, ported from /home/sinep/Sinlate/config.py so the two
tools behave identically: unknown keys in the on-disk file are dropped, a corrupt file degrades
to defaults with a warning instead of crashing, and writes go through a temp file + os.replace.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "sinribe"
CONFIG_PATH = CONFIG_DIR / "config.json"
TOKEN_PATH = CONFIG_DIR / "hf_token"
CACHE_DIR = Path.home() / ".cache" / "sinribe"
JOBS_DIR = CACHE_DIR / "jobs"
DOWNLOADS_DIR = CACHE_DIR / "downloads"
# CTranslate2 conversions of models that have no ready-made faster-whisper build. Kept out of the
# HuggingFace cache because they are not HuggingFace artefacts — they are locally converted, and a
# `huggingface-cli delete-cache` would otherwise throw away work that takes minutes to redo.
MODELS_DIR = CACHE_DIR / "models"

# German fine-tunes of whisper. Vanilla large-v3 is multilingual and treats German as one language
# among ninety-nine; these were trained on German alone, which is what a Bavarian-accented room
# recording needs. Both are converted locally by tools/convert_german_models.sh — the upstream
# repos ship transformers weights, not CTranslate2 ones.
LOCAL_MODELS = {
    "large-v3-german": "whisper-large-v3-german-ct2",
    "large-v3-turbo-german": "whisper-large-v3-turbo-german-ct2",
}

# Models that cannot be given a hotword prompt. `large-v3-german` returns an EMPTY transcript for
# the whole recording when one is supplied — measured on two different files, with hotword strings
# as short as a single letter, while the same model with no hotwords transcribes normally. Its
# tokenizer and vocabulary match the base model exactly, so this is the fine-tune itself having
# lost the ability to condition on a prefix, not a conversion fault. `large-v3-turbo-german` from
# the same publisher is unaffected.
MODELS_WITHOUT_HOTWORDS = frozenset({"large-v3-german"})


def supports_hotwords(name: str) -> bool:
    return name not in MODELS_WITHOUT_HOTWORDS

# "auto" lets the quality setting choose; everything else is an explicit override.
ASR_MODELS = ["auto", "large-v3", "large-v3-german", "large-v3-turbo-german", "medium.en", "small"]


def model_dir(name: str) -> Path | None:
    """Where a locally converted model lives, or None if `name` is not one of ours."""
    sub = LOCAL_MODELS.get(name)
    return (MODELS_DIR / sub) if sub else None


def model_available(name: str) -> bool:
    """Is this model ready to use? Bare whisper sizes are assumed cached; ours must be on disk."""
    d = model_dir(name)
    return (d / "model.bin").exists() if d else True


def resolve_model(name: str) -> str:
    """Model name as faster-whisper wants it — a directory for our conversions, else the size.

    `WhisperModel` treats an existing directory as the model and anything else as a name to fetch,
    so this is the whole of the local-model plumbing.
    """
    d = model_dir(name)
    return str(d) if d else name

DIAR_PIPELINES = [
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",
]


def _hf_cache() -> Path:
    """Where huggingface_hub keeps its snapshots, honouring the usual environment overrides."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def diar_available(name: str) -> bool:
    """Is this diarization pipeline actually in the local cache?

    Workers run with `HF_HUB_OFFLINE=1`, so a pipeline that was never downloaded cannot be
    fetched at job time — it fails several minutes in, after the audio has been decoded. Both
    entries above are offered in the UI, but `speaker-diarization-3.1` is gated behind a separate
    licence acceptance and is routinely absent; selecting it used to be a working choice right up
    until the job died. Checked so the UI can grey it out instead.
    """
    snapshots = _hf_cache() / f"models--{name.replace('/', '--')}" / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())

# Subset of whisper's languages worth surfacing in the UI; "auto" lets whisper detect.
LANGUAGES = [
    ("auto", "Auto-detect"),
    ("de", "German"),
    ("en", "English"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("it", "Italian"),
    ("nl", "Dutch"),
    ("pt", "Portuguese"),
    ("pl", "Polish"),
    ("ru", "Russian"),
    ("tr", "Turkish"),
    ("zh", "Chinese"),
    ("ja", "Japanese"),
]

DEFAULTS: dict = {
    # paths
    "output_dir": str(Path.home() / "Sinribe-Transcripts"),
    "last_input_dir": str(Path.home()),
    # ASR
    # "auto" lets the accuracy slider pick the model, which is what makes the slider a single
    # honest control instead of a setting you have to pair correctly with the dropdown.
    "asr_model": "auto",
    "language": "auto",
    "vad_filter": True,
    # The accuracy slider, as a target realtime factor: lower is slower and more accurate.
    # Everything it implies (model, beam width, VAD padding, audio conditioning) lives in
    # presets.py next to the WER each rung actually measured. The default is the rung with the
    # best measured accuracy-per-minute, not the slowest one on offer.
    "speed_target": 10,
    # Realtime factors actually observed on THIS machine, keyed by rung, updated after every
    # successful job. The shipped figures in presets.py came off one benchmark run and proved
    # unreproducible — the same settings measured 50× and 20× on different days, because the GPU
    # is shared with Ollama and whatever else is resident. Rather than assert a number the user
    # can catch being wrong, the estimate corrects itself from their own runs.
    "observed_rtf": {},
    # Names and jargon to bias the recogniser toward, comma-separated. Whisper reliably mangles
    # proper nouns it has no reason to expect — "Fujitsu" came out as *fiuzi*, *jitze* and
    # *future service* in one interview — and this is the supported fix.
    "hotwords": "",
    # diarization
    "diar_pipeline": "pyannote/speaker-diarization-community-1",
    "speaker_mode": "auto",  # auto | exact | range
    "num_speakers": 2,
    "min_speakers": 1,
    "max_speakers": 6,
    # Re-check each sentence against the speakers' voice prints and correct the diarization where
    # the acoustic match is decisive. Costs a few seconds; see workers/refine_worker.py.
    "refine_speakers": True,
    # Cosine gap to the runner-up needed to overrule diarization. Below ~0.1 the two voices are
    # indistinguishable on that span and the diarization's wider context is the better guess.
    "refine_margin": 0.15,
    # A span shorter than this is too little voice to identify; backchannels ("Okay.", "Genau.")
    # stay with whatever the diarization said. Measured margins on real audio are bimodal —
    # confident corrections sit at 0.3+, coin flips under 0.1 — so 0.15 falls in the empty gap.
    "refine_min_seconds": 0.6,
    # Decide attribution one sentence at a time rather than one word at a time.
    "sentence_atomic": True,
    # merge / formatting
    "turn_gap_s": 1.5,
    "max_turn_chars": 1200,
    "flicker_min_words": 3,
    "orphan_word_window_s": 2.0,
    # outputs
    # Append the list of passages the recogniser was least sure of. Cheap, and it turns
    # proofreading a transcript from "re-listen to all of it" into "check these forty spots".
    "review_section": True,
    # How many of those passages to list. 0 means all of them, which is what makes the section a
    # checklist rather than a sample — a hard 65-minute interview flags 363 and every one is a
    # place worth an ear. Set a positive number to keep only that many least-confident entries.
    "review_max_spans": 0,
    "write_json": True,
    "write_srt": True,
    "write_vtt": True,
    # optional LLM enrichment (Ollama, offline)
    "llm_enrich": False,
    "llm_url": "http://127.0.0.1:11435",
    "llm_model": "gemma3:4b",
    "llm_timeout": 180,
    # URL download (yt-dlp)
    "last_url": "",
    # Browser to lift cookies from for age-restricted or members-only media, e.g. "firefox"
    # or "chrome". Empty means no cookies, which is fine for ordinary public podcasts.
    "yt_cookies_from_browser": "",
    "keep_downloads": True,
    # housekeeping
    "keep_decoded_wav": False,
    "cache_days": 14,
    "download_cache_days": 14,
}


def _migrate(user: dict, cfg: dict) -> None:
    """Bring a pre-slider config forward."""
    if "speed_target" not in user and "asr_mode" in user:
        # The old binary control: "accurate" was the sequential path, "fast" the batched one.
        # Map each onto the rung that still does that, so an upgrade changes the wording and not
        # the behaviour. Taken from presets so a recalibrated ladder cannot leave this behind
        # pointing at a rung that no longer exists.
        from .presets import RUNGS
        want_fast = str(user["asr_mode"]) != "accurate"
        match = [r for r in RUNGS
                 if (r.settings()["asr_mode"] == "fast") == want_fast
                 and r.model == "large-v3"]
        if match:
            cfg["speed_target"] = min(r.target_rtf for r in match)
    if "speed_target" not in user and user.get("asr_model") == "large-v3":
        # large-v3 was the default AND the only serious option, so a saved "large-v3" records no
        # preference — it is just what the app came with. Treated as a pin it would quietly
        # override the quality slider's model choice and hand an existing user none of the
        # accuracy work. A genuine preference can still be set explicitly now that there is a
        # choice to make.
        cfg["asr_model"] = "auto"


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            user = json.loads(CONFIG_PATH.read_text())
            cfg.update({k: v for k, v in user.items() if k in DEFAULTS})
            _migrate(user, cfg)
        except Exception as e:  # noqa: BLE001 - a broken config must never block startup
            print(f"[sinribe] WARNING: could not read {CONFIG_PATH}: {e!r}; using defaults",
                  flush=True)
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    payload = {k: cfg[k] for k in DEFAULTS if k in cfg}
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, CONFIG_PATH)


def hf_token() -> str | None:
    """Read the HuggingFace token written by install.sh (chmod 600, never in the repo)."""
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env.strip()
    try:
        tok = TOKEN_PATH.read_text().strip()
        return tok or None
    except OSError:
        return None
