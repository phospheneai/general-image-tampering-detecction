#!/usr/bin/env bash

# ============================================================
# Authenta General Image Forgery
# One-shot local / Lambda setup
# ============================================================
#
# Run after a Lambda session restart:
#
#   bash scripts/setup.sh
#
# This script:
#
#   1. Installs the project dependencies.
#   2. Installs the Authenta package in editable mode.
#   3. Ensures the required local artifact directories exist.
#
# The pretrained DINOv3 ViT-L/16 model is NOT downloaded by
# this script.
#
# DINOv3 pretrained files are kept outside Git and are supplied
# separately as a Hugging Face model artifact:
#
#   config.json
#   model.safetensors
#
# SageMaker receives that artifact through the `artifacts`
# input channel.
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


# ============================================================
# 1. Dependencies
# ============================================================

echo
echo "============================================================"
echo "[1/3] Installing dependencies"
echo "============================================================"

bash "$SCRIPT_DIR/install_deps.sh"

echo "      dependencies OK"


# ============================================================
# 2. Artifact directories
# ============================================================

echo
echo "============================================================"
echo "[2/3] Preparing artifact directories"
echo "============================================================"

DINO_DIR="$ROOT/artifacts/mirror/dinov3-vitl16"

mkdir -p "$DINO_DIR"

echo "      DINOv3 artifact directory:"
echo "      $DINO_DIR"

echo
echo "      Expected files:"
echo "        - config.json"
echo "        - model.safetensors"


# ============================================================
# 3. Done
# ============================================================

echo
echo "============================================================"
echo "[3/3] Setup complete"
echo "============================================================"

echo "      project root : $ROOT"
echo "      DINOv3 path  : $DINO_DIR"

echo
echo "      The DINOv3 pretrained weights are external to Git."
echo "      Place the Hugging Face model files in the directory"
echo "      above when local model execution is required."

echo