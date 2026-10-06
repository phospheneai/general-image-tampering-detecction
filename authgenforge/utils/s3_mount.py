"""
Mount an S3 prefix as a read-only local folder, so the dataset is read like
files on a drive instead of being streamed by the training code.

Uses Mountpoint for Amazon S3 (`mount-s3`, install with
scripts/install_mountpoint.sh). Credentials come from the usual AWS chain
(`aws configure`, AWS_PROFILE, instance role).

    python -m authgenforge.utils.s3_mount \\
        s3://authenta-data-rnd/image-tampering-detection/ \\
        ~/data/s3/image-tampering-detection

    python -m authgenforge.utils.s3_mount --unmount ~/data/s3/image-tampering-detection

A training yml then points its dataroots at folders under the mount point.
train.py does the mount itself from the yml's `s3_mount:` block. On
SageMaker none of this is used: the job's input channels are already mounted
at /opt/ml/input/data/<channel>/.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

INSTALL_HINT = "Fix: bash scripts/install_mountpoint.sh"


class S3MountError(RuntimeError):
    pass


def _split_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise S3MountError(f"not an s3:// URI: {uri!r}")
    bucket, _, prefix = uri[len("s3://"):].partition("/")
    if not bucket:
        raise S3MountError(f"no bucket in {uri!r}")
    prefix = prefix.strip("/")
    return bucket, (prefix + "/" if prefix else "")


def _expand(path) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


def is_mounted(mount_point) -> bool:
    return os.path.ismount(_expand(mount_point))


def mount_s3(
    uri: str,
    mount_point,
    cache_dir=None,
    max_cache_gb: float | None = None,
    region: str | None = None,
) -> Path:
    """
    Mount `uri` read-only at `mount_point` and return the mount point.
    A no-op when something is already mounted there.

    cache_dir: optional local folder where Mountpoint keeps the blocks it
        has read, so reading them again does not go back to S3.
    max_cache_gb: cap on that folder; Mountpoint evicts beyond it.
    """
    mount_point = _expand(mount_point)

    if os.path.ismount(mount_point):
        return mount_point

    binary = shutil.which("mount-s3")
    if not binary:
        raise S3MountError(f"mount-s3 is not installed.\n{INSTALL_HINT}")

    bucket, prefix = _split_uri(uri)
    mount_point.mkdir(parents=True, exist_ok=True)
    if any(mount_point.iterdir()):
        raise S3MountError(f"{mount_point} is not empty — refusing to mount over it")

    cmd = [binary, bucket, str(mount_point), "--read-only"]
    if prefix:
        cmd += ["--prefix", prefix]
    if region:
        cmd += ["--region", region]
    if cache_dir:
        cache_dir = _expand(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cmd += ["--cache", str(cache_dir)]
        if max_cache_gb:
            cmd += ["--max-cache-size", str(int(max_cache_gb * 1024))]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.ismount(mount_point):
        raise S3MountError(
            f"could not mount {uri} at {mount_point}:\n"
            f"{(result.stderr or result.stdout).strip()}"
        )
    return mount_point


def unmount(mount_point) -> None:
    mount_point = _expand(mount_point)
    if not os.path.ismount(mount_point):
        return
    tool = shutil.which("fusermount3") or shutil.which("fusermount")
    cmd = [tool, "-u", str(mount_point)] if tool else ["umount", str(mount_point)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise S3MountError(f"could not unmount {mount_point}: {result.stderr.strip()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("uri", nargs="?", help="s3://bucket/prefix/ to mount")
    ap.add_argument("mount_point", nargs="?", help="local folder to mount it at")
    ap.add_argument("--unmount", metavar="MOUNT_POINT", default=None)
    ap.add_argument("--cache-dir", default=None, help="keep read blocks here (optional)")
    ap.add_argument("--max-cache-gb", type=float, default=None)
    ap.add_argument("--region", default=None)
    args = ap.parse_args()

    try:
        if args.unmount:
            unmount(args.unmount)
            print(f"unmounted {_expand(args.unmount)}")
            return
        if not (args.uri and args.mount_point):
            ap.error("give an s3:// URI and a mount point (or --unmount MOUNT_POINT)")
        path = mount_s3(args.uri, args.mount_point, args.cache_dir, args.max_cache_gb, args.region)
        print(f"{args.uri} mounted at {path}")
    except S3MountError as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
