#!/usr/bin/env bash

# ============================================================
# Authenta General Image Forgery
# DINOv3 ViT-L/16 artifact downloader
# ============================================================
#
# Downloads the official Hugging Face DINOv3 ViT-L/16 model
# into:
#
#   artifacts/mirror/dinov3-vitl16/
#
# Model:
#
#   facebook/dinov3-vitl16-pretrain-lvd1689m
#
# The downloaded directory is expected to contain the Hugging
# Face model files, including:
#
#   config.json
#   model.safetensors
#
# The pretrained model is intentionally NOT committed to Git.
#
# Run from the repository root:
#
#   bash scripts/download_artifacts.sh
#
# ============================================================

set -euo pipefail


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

ROOT="$(
    dirname "$SCRIPT_DIR"
)"

DEST="$ROOT/artifacts/mirror/dinov3-vitl16"

MODEL_ID="facebook/dinov3-vitl16-pretrain-lvd1689m"


# ============================================================
# Prepare destination
# ============================================================

mkdir -p "$DEST"

echo
echo "============================================================"
echo "[artifacts] DINOv3 ViT-L/16"
echo "============================================================"
echo "[artifacts] model      : $MODEL_ID"
echo "[artifacts] destination: $DEST"
echo


# ============================================================
# Check for an existing complete artifact
# ============================================================

if [[ -f "$DEST/config.json" && \
      -f "$DEST/model.safetensors" ]]; then

    echo "[skip] DINOv3 ViT-L/16 artifact already exists."
    echo "       $DEST"

    exit 0
fi


# ============================================================
# Check Hugging Face CLI
# ============================================================

if ! command -v hf >/dev/null 2>&1; then

    echo "[error] Hugging Face CLI 'hf' was not found."
    echo
    echo "Install it with:"
    echo
    echo "    python -m pip install huggingface_hub"
    echo

    exit 1
fi


# ============================================================
# Download
# ============================================================

echo "[download] downloading Hugging Face model..."
echo

hf download \
    "$MODEL_ID" \
    --local-dir "$DEST"


# ============================================================
# Validate artifact
# ============================================================

echo
echo "[validate] checking downloaded files..."

if [[ ! -f "$DEST/config.json" ]]; then
    echo "[error] config.json was not downloaded."
    exit 1
fi

if [[ ! -f "$DEST/model.safetensors" ]]; then
    echo "[error] model.safetensors was not downloaded."
    exit 1
fi


# ============================================================
# Complete
# ============================================================

echo
echo "============================================================"
echo "[done] DINOv3 ViT-L/16 artifact downloaded successfully."
echo "============================================================"
echo
echo "Model directory:"
echo "    $DEST"
echo
echo "Files:"
echo "    config.json"
echo "    model.safetensors"
echo