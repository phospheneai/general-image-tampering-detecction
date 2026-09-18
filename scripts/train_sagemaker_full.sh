#!/usr/bin/env bash

# ============================================================
# Authenta General Image Forgery
# Full SageMaker training wrapper
# ============================================================
#
# Thin wrapper around:
#
#   sagemaker/launch.py
#
# It provides the larger-run defaults while allowing any
# additional launch.py arguments to be passed through.
#
# Examples:
#
#   bash scripts/train_sagemaker_full.sh \
#       --run-name full-forensics-v1
#
#   bash scripts/train_sagemaker_full.sh \
#       --run-name full-forensics-v1 \
#       --no-spot
#
#   bash scripts/train_sagemaker_full.sh \
#       --run-name full-forensics-v1 \
#       --end-epoch 2
#
# AWS credentials are expected to come from the normal AWS
# credential chain:
#
#   aws configure
#   AWS_PROFILE
#   environment variables
#   IAM / instance credentials
#
# The SageMaker execution role is supplied through:
#
#   --role
#
# or:
#
#   SAGEMAKER_ROLE
#
# Region, output bucket, and input channels are defined in:
#
#   sagemaker/config/normal/train_forensics.yml
#
# ============================================================

set -euo pipefail


# ============================================================
# Repository root
# ============================================================

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

ROOT="$(
    dirname "$SCRIPT_DIR"
)"

cd "$ROOT"


# ============================================================
# SageMaker execution role
# ============================================================

ROLE="${SAGEMAKER_ROLE:-}"


# ============================================================
# Fail early on invalid AWS credentials
# ============================================================

case " $* " in
    *" --dry-run "*)
        ;;
    *)
        aws sts get-caller-identity >/dev/null
        ;;
esac


# ============================================================
# Build launch arguments
# ============================================================

LAUNCH_ARGS=(
    --config
    "config/normal/train_forensics.yml"

    --volume-size
    "200"

    --max-run-hours
    "72"

    --max-wait-hours
    "94"
)


# ------------------------------------------------------------
# Add execution role when supplied
# ------------------------------------------------------------

if [[ -n "$ROLE" ]]; then
    LAUNCH_ARGS+=(
        --role
        "$ROLE"
    )
fi


# ============================================================
# Execute SageMaker launcher
# ============================================================

set -x

exec python sagemaker/launch.py \
    "${LAUNCH_ARGS[@]}" \
    "$@"