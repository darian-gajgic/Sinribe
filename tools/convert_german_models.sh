#!/usr/bin/env bash
# Install the German Whisper models Sinribe can use.
#
# The upstream repos publish transformers weights; faster-whisper needs CTranslate2 ones, so they
# are converted here once and cached in ~/.cache/sinribe/models/. Roughly 4.7 GB downloaded and
# 4.7 GB kept, and a few minutes on this machine.
#
# The conversion runs in a THROWAWAY venv with CPU-only torch, and that is not a detail. Sinribe's
# .venv holds the ctranslate2 / CUDA-12 stack; PyPI torch brings CUDA 13, and cuDNN 12 and 13 share
# a SONAME. Putting both in one virtualenv is precisely the breakage the two-venv split exists to
# prevent, so never "simplify" this by installing transformers into .venv.
#
#   tools/convert_german_models.sh              both models
#   tools/convert_german_models.sh turbo        just the fast one
set -euo pipefail

VENV="${TMPDIR:-/tmp}/sinribe-ct2conv"
OUT="${SINRIBE_MODELS_DIR:-$HOME/.cache/sinribe/models}"
WHICH="${1:-all}"

declare -A REPOS=(
    [full]="primeline/whisper-large-v3-german:whisper-large-v3-german-ct2"
    [turbo]="primeline/whisper-large-v3-turbo-german:whisper-large-v3-turbo-german-ct2"
)

case "$WHICH" in
    all)   TARGETS=(full turbo) ;;
    full)  TARGETS=(full) ;;
    turbo) TARGETS=(turbo) ;;
    *) echo "usage: $0 [all|full|turbo]" >&2; exit 2 ;;
esac

command -v uv >/dev/null || { echo "ERROR: uv is not installed" >&2; exit 1; }
mkdir -p "$OUT"

cleanup() { rm -rf "$VENV"; }
trap cleanup EXIT

echo "==> preparing conversion environment (throwaway, CPU-only torch)"
rm -rf "$VENV"
uv venv --python 3.12 "$VENV"
uv pip install --quiet --python "$VENV/bin/python" \
    --index-strategy unsafe-best-match \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch transformers ctranslate2==4.8.0

for target in "${TARGETS[@]}"; do
    spec="${REPOS[$target]}"
    repo="${spec%%:*}"
    dest="$OUT/${spec##*:}"

    if [ -f "$dest/model.bin" ] && [ -f "$dest/tokenizer.json" ]; then
        echo "==> $target already installed at $dest"
        continue
    fi

    echo "==> converting $repo"
    # Written to a .part directory and moved into place only on success, so an interrupted
    # download can never leave a half-model that Sinribe would then try to load.
    rm -rf "$dest.part"
    HF_HUB_OFFLINE=0 "$VENV/bin/ct2-transformers-converter" \
        --model "$repo" \
        --output_dir "$dest.part" \
        --copy_files preprocessor_config.json \
        --quantization float16 \
        --force

    # Some of these repos ship only the slow tokenizer's vocab.json/merges.txt. faster-whisper
    # loads tokenizer.json, so build the fast one rather than depending on the repo having it.
    HF_HUB_OFFLINE=0 "$VENV/bin/python" - "$repo" "$dest.part" <<'PY'
import sys
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(sys.argv[1]).save_pretrained(sys.argv[2])
PY

    rm -rf "$dest"
    mv "$dest.part" "$dest"
    echo "==> installed $(du -sh "$dest" | cut -f1) at $dest"
done

echo
echo "Done. Pick the model in Sinribe's options, or leave it on 'auto'."
