#!/usr/bin/env bash
# Install Mountpoint for Amazon S3 (`mount-s3`), which mounts the dataset
# bucket as a read-only local folder — see authgenforge/utils/s3_mount.py.
# Needed for local runs only; SageMaker mounts its input channels itself.
#
#   bash scripts/install_mountpoint.sh
#
# Ubuntu/Debian, x86_64 or arm64. Needs sudo (the package pulls in FUSE).
set -euo pipefail

if command -v mount-s3 >/dev/null 2>&1; then
  echo "[install] already installed: $(mount-s3 --version)"
  exit 0
fi

case "$(uname -m)" in
  x86_64)  ARCH=x86_64 ;;
  aarch64) ARCH=arm64 ;;
  *) echo "[install] unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
curl -fsSL -o "$TMP/mount-s3.deb" \
  "https://s3.amazonaws.com/mountpoint-s3-release/latest/${ARCH}/mount-s3.deb"
sudo apt-get install -y "$TMP/mount-s3.deb"
echo "[install] $(mount-s3 --version)"
