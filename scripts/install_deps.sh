#!/bin/bash

# ============================================================
# Authenta General Image Forgery
# Local / Lambda dependency installation
# ============================================================
#
# This script installs the Python dependencies required by the
# Authenta general image forgery segmentation project.
#
# Lambda Stack already provides:
#   - PyTorch
#   - torchvision
#   - CUDA
#
# Run after a Lambda session restart when required:
#
#   bash scripts/install_deps.sh
#
# SageMaker dependencies are maintained separately in:
#
#   sagemaker/requirements.txt
#
# ============================================================

set -e


# ============================================================
# Upgrade pip
# ============================================================

python -m pip install --quiet --upgrade pip


# ============================================================
# Core project dependencies
# ============================================================

# NumPy remains pinned to the 1.x ABI because the Lambda Stack
# PyTorch/torchvision environment is built against NumPy 1.x.
#
# OpenCV is intentionally kept below the releases that require
# NumPy 2.x.

python -m pip install \
    numpy==1.26.4 \
    transformers==4.57.1 \
    tokenizers==0.22.2 \
    huggingface-hub==0.36.2 \
    peft==0.19.1 \
    accelerate==1.10.1 \
    datasets==5.0.0 \
    torchdata==0.11.0 \
    mosaicml-streaming==0.13.0 \
    opencv-python-headless==4.11.0.86 \
    pillow-heif==1.7.0 \
    pymupdf==1.28.0


# ============================================================
# Project utilities
# ============================================================

python -m pip install \
    pillow \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm \
    pyyaml


# ============================================================
# Install the Authenta package
# ============================================================

python -m pip install --quiet -e "$(dirname "$0")/.."


# ============================================================
# Complete
# ============================================================

echo
echo "============================================================"
echo "Authenta dependencies installed successfully."
echo "============================================================"