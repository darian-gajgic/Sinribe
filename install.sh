#!/usr/bin/env bash
# Sinribe installer — no sudo, no system packages, no driver/CUDA/kernel changes.
#
#   ./install.sh --hf-token hf_xxxxxxxx     first run (downloads the pyannote models)
#   ./install.sh                            rebuild venvs, reuse the stored token
#
# Creates TWO virtualenvs on purpose: ctranslate2 needs CUDA 12 (libcublas.so.12) while torch
# 2.13 from PyPI is a CUDA 13 build. They cannot share an environment.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV="${UV:-$HOME/.local/bin/uv}"
CONFIG_DIR="$HOME/.config/sinribe"
TOKEN_FILE="$CONFIG_DIR/hf_token"
BIN_DIR="$HOME/.local/bin"
APPS_DIR="$HOME/.local/share/applications"

HF_TOKEN_ARG=""
SKIP_MODELS=0
SKIP_VENVS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-token) HF_TOKEN_ARG="${2:-}"; shift 2 ;;
    --skip-models) SKIP_MODELS=1; shift ;;
    --skip-venvs) SKIP_VENVS=1; shift ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ -x "$UV" ]] || die "uv not found at $UV — install it or set UV=/path/to/uv"
command -v ffmpeg  >/dev/null || die "ffmpeg not found (apt install ffmpeg)"
command -v ffprobe >/dev/null || die "ffprobe not found (apt install ffmpeg)"

# ---------------------------------------------------------------- token
mkdir -p "$CONFIG_DIR"; chmod 700 "$CONFIG_DIR"
if [[ -n "$HF_TOKEN_ARG" ]]; then
  printf '%s' "$HF_TOKEN_ARG" > "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
  say "stored HuggingFace token in $TOKEN_FILE (chmod 600)"
fi
[[ -s "$TOKEN_FILE" ]] || die "no HuggingFace token. pyannote's models are gated:
  1. create an account at https://huggingface.co/join
  2. accept the licence at https://huggingface.co/pyannote/segmentation-3.0
  3. accept the licence at https://huggingface.co/pyannote/speaker-diarization-community-1
  4. make a READ token at https://huggingface.co/settings/tokens
  5. re-run:  ./install.sh --hf-token hf_xxxxxxxx"

# ---------------------------------------------------------------- venvs
if [[ "$SKIP_VENVS" -eq 0 ]]; then
  say "building ASR + GUI venv (.venv, CUDA 12 stack)"
  "$UV" venv --python 3.12 "$HERE/.venv"
  "$UV" pip install --python "$HERE/.venv/bin/python" -r "$HERE/requirements-asr.txt"

  say "building diarization venv (.venv-diar, CUDA 13 / torch stack — this one is ~5 GB)"
  "$UV" venv --python 3.12 "$HERE/.venv-diar"
  "$UV" pip install --python "$HERE/.venv-diar/bin/python" -r "$HERE/requirements-diar.txt"
else
  say "skipping venv build (--skip-venvs)"
  [[ -x "$HERE/.venv/bin/python" ]]      || die ".venv is missing; re-run without --skip-venvs"
  [[ -x "$HERE/.venv-diar/bin/python" ]] || die ".venv-diar is missing; re-run without --skip-venvs"
fi

# ---------------------------------------------------------------- models
if [[ "$SKIP_MODELS" -eq 0 ]]; then
  say "downloading pyannote models (one time; everything is offline afterwards)"
  HF_HUB_OFFLINE=0 "$HERE/.venv-diar/bin/python" - <<'PY'
import os, sys
tok = open(os.path.expanduser("~/.config/sinribe/hf_token")).read().strip()
from pyannote.audio import Pipeline
name = "pyannote/speaker-diarization-community-1"
try:
    p = Pipeline.from_pretrained(name, token=tok)
except Exception as e:
    print(f"  FAILED to fetch {name}: {type(e).__name__}: {e}", file=sys.stderr)
    print("  -> accept the licence on huggingface.co with this token's account.", file=sys.stderr)
    sys.exit(1)
if p is None:
    print(f"  {name} returned None — the licence has not been accepted for this account.",
          file=sys.stderr)
    sys.exit(1)
print(f"  ok: {name}")
PY

  # Fetched here because nothing else will: the workers force HF_HUB_OFFLINE=1 (pipeline._asr_env),
  # so a model missing at this point makes the first transcription fail to load it.
  say "downloading the whisper model (large-v3, ~3 GB, one time)"
  HF_HUB_OFFLINE=0 "$HERE/.venv/bin/python" - <<'PY'
import sys
from faster_whisper import download_model
try:
    p = download_model("large-v3")
except Exception as e:
    print(f"  FAILED to fetch large-v3: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
print(f"  ok: large-v3 at {p}")
PY

  # The default quality rung votes with this model; without it that pass is skipped and the
  # default quietly loses accuracy. Optional, so a failure here warns instead of aborting.
  if [[ -f "$HOME/.cache/sinribe/models/whisper-large-v3-turbo-german-ct2/model.bin" ]]; then
    say "German turbo model already installed"
  else
    say "converting the German turbo model the default setting votes with (one time, 1.6 GB kept)"
    PATH="$(dirname "$UV"):$PATH" "$HERE/tools/convert_german_models.sh" turbo \
      || say "WARNING: German turbo model not installed; the default setting will run 2 of its 3 passes. Retry with: tools/convert_german_models.sh turbo"
  fi
fi

# ---------------------------------------------------------------- launchers
say "installing launcher and desktop entry"
chmod +x "$HERE/sinribe-run" "$HERE/sinribe-eval" "$HERE/tools/convert_german_models.sh"
mkdir -p "$BIN_DIR" "$APPS_DIR"
ln -sfn "$HERE/sinribe-run" "$BIN_DIR/sinribe"
ln -sfn "$HERE/sinribe-eval" "$BIN_DIR/sinribe-eval"

sed "s|^Exec=.*|Exec=$HERE/sinribe-run|" "$HERE/sinribe.desktop" > "$APPS_DIR/sinribe.desktop"
chmod 644 "$APPS_DIR/sinribe.desktop"
if [[ -d "$HOME/Desktop" ]]; then
  cp -f "$APPS_DIR/sinribe.desktop" "$HOME/Desktop/sinribe.desktop"
  chmod +x "$HOME/Desktop/sinribe.desktop"
  # GNOME refuses to launch a desktop file it does not consider trusted.
  gio set "$HOME/Desktop/sinribe.desktop" metadata::trusted true 2>/dev/null || true
fi
update-desktop-database "$APPS_DIR" 2>/dev/null || true

mkdir -p "$(grep -oP '(?<="output_dir": ")[^"]*' "$CONFIG_DIR/config.json" 2>/dev/null \
           || echo "$HOME/Sinribe-Transcripts")" 2>/dev/null || true

say "done"
cat <<EOF

  Launch:      sinribe                      (or the Sinribe icon)
  Headless:    sinribe path/to/audio.mp3 -o ~/Transcripts

  If '$BIN_DIR' is not on your PATH, add it to ~/.bashrc:
      export PATH="\$HOME/.local/bin:\$PATH"
EOF
