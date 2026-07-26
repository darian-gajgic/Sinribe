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

# Whisper models already cached on this box (no download needed for any of these).
ASR_MODELS = ["large-v3", "medium.en", "small"]

DIAR_PIPELINES = [
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",
]

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
    "asr_model": "large-v3",
    "language": "auto",
    "beam_size": 5,
    "vad_filter": True,
    # diarization
    "diar_pipeline": "pyannote/speaker-diarization-community-1",
    "speaker_mode": "auto",  # auto | exact | range
    "num_speakers": 2,
    "min_speakers": 1,
    "max_speakers": 6,
    # merge / formatting
    "turn_gap_s": 1.5,
    "max_turn_chars": 1200,
    "flicker_min_words": 3,
    "orphan_word_window_s": 2.0,
    # outputs
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


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            user = json.loads(CONFIG_PATH.read_text())
            cfg.update({k: v for k, v in user.items() if k in DEFAULTS})
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
