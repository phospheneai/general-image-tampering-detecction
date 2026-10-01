#!/usr/bin/env python
"""
Train the forgery-segmentation model end to end on this machine's GPU.

    python train.py                  # full run: configs/local/train_forensics.yml
    python train.py --smoke          # few-minute end-to-end check on the smoke slice
    python train.py --config X.yml --epochs N

One command does everything, in order:

  1. GPU preflight   fail fast (with the fix) if CUDA is unusable — no driver
                     module, or a torch build without kernels for this GPU —
                     instead of silently training on the CPU.
  2. AWS preflight   s3:// dataroots need working credentials.
  3. Backbone        DINOv3 ViT-L/16 config.json + model.safetensors at the
                     config's model_path, downloaded from S3 if missing.
  4. Data            processed-v1 MDS streamed from S3 into a bounded local
                     cache (shard-block shuffling, next shards prefetched).
  5. Resume          if <checkpoints>/<name>/latest_checkpoint.pth exists,
                     continue from it — same logic as the SageMaker job
                     (sagemaker/train.py), down to the mid-epoch loader
                     position.
  6. Train           load_pipeline_from_yml + trainer.train_model, exactly
                     what the SageMaker entry point runs.

Interrupt it (Ctrl+C, crash, reboot) and run the same command again to
continue. The SageMaker pipeline is separate: see infra/README.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent

DEFAULT_CONFIG = REPO / "configs" / "local" / "train_forensics.yml"
SMOKE_CONFIG = REPO / "configs" / "local" / "smoke.yml"
DEFAULT_CACHE_DIR = Path.home() / "data" / "mds_cache"

# Where the SageMaker job gets the backbone — the single source of truth.
SAGEMAKER_CONFIG = REPO / "sagemaker" / "config" / "normal" / "train_forensics.yml"

SHARD_GB = 0.54         # processed-v1 shards are written at <= 512 MiB
DISK_HEADROOM_GB = 30   # left free for checkpoints (~1.3 GB each) and the OS


def log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"\n[train] ERROR: {msg}\n", file=sys.stderr, flush=True)
    sys.exit(2)


# ============================================================
# 1. GPU
# ============================================================

def check_gpu(allow_cpu: bool) -> None:
    import torch

    if not torch.cuda.is_available():
        reason = "torch.cuda.is_available() is False"
        fix = "check `nvidia-smi`"
        if platform.system() == "Linux" and not Path("/proc/driver/nvidia/version").exists():
            kernel = platform.release()
            reason = f"the NVIDIA kernel module is not loaded (kernel {kernel})"
            fix = (
                "install the driver module for the running kernel, e.g.\n"
                f"    sudo apt install linux-modules-nvidia-580-open-{kernel}\n"
                "    sudo modprobe nvidia && nvidia-smi"
            )
        elif torch.version.cuda is None:
            reason = f"torch {torch.__version__} is a CPU-only build"
            fix = "install a CUDA build of torch"
        if allow_cpu:
            log(f"WARNING: no usable GPU ({reason}) — running on CPU (--allow-cpu)")
            return
        fail(f"no usable GPU: {reason}.\nFix: {fix}\n(--allow-cpu runs on the CPU anyway — very slow.)")

    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    supported = torch.cuda.get_arch_list()
    cap = major * 10 + minor

    def runs(a: str) -> bool:
        kind, _, num = a.partition("_")
        n = int("".join(c for c in num if c.isdigit()) or 0)
        if kind == "sm":       # cubin: same major, minor <= the device's
            return n // 10 == major and n <= cap
        return n <= cap        # compute_NN: PTX, JIT-compiled for newer GPUs

    if not any(runs(a) for a in supported):
        fail(
            f"{torch.cuda.get_device_name(0)} is {arch}, but torch {torch.__version__} "
            f"(CUDA {torch.version.cuda}) has kernels only for {' '.join(supported)}.\n"
            "Fix: install a torch build for this GPU — for Blackwell (sm_120):\n"
            "    pip install torch==2.7.1 torchvision==0.22.1 "
            "--index-url https://download.pytorch.org/whl/cu128"
        )

    props = torch.cuda.get_device_properties(0)
    log(
        f"GPU {props.name} ({arch}, {props.total_memory / 2**30:.0f} GiB) · "
        f"torch {torch.__version__} · CUDA {torch.version.cuda}"
    )


# ============================================================
# 2-4. AWS, backbone, cache
# ============================================================

def _is_s3(path) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def check_aws() -> None:
    try:
        import boto3

        ident = boto3.client("sts").get_caller_identity()
    except Exception as e:  # no credentials, expired token, no network
        fail(f"the data is on S3 but AWS credentials don't work: {e}\nFix: `aws configure` (or AWS_PROFILE).")
    log(f"AWS {ident['Arn']}")


def backbone_s3_uri() -> str:
    cfg = yaml.safe_load(SAGEMAKER_CONFIG.read_text(encoding="utf-8"))
    return cfg["sagemaker"]["inputs"]["artifacts"]["uri"]


def ensure_backbone(model_dir: Path) -> None:
    needed = ["config.json", "model.safetensors"]
    missing = [f for f in needed if not (model_dir / f).is_file()]
    if not missing:
        log(f"backbone {model_dir}")
        return

    uri = backbone_s3_uri()
    log(f"backbone missing {missing} — downloading from {uri}")
    model_dir.mkdir(parents=True, exist_ok=True)
    import boto3
    from boto3.s3.transfer import TransferConfig

    bucket, _, prefix = uri[len("s3://"):].partition("/")
    s3 = boto3.client("s3")
    for name in missing:
        tmp = model_dir / f"{name}.part"
        try:
            s3.download_file(bucket, prefix.rstrip("/") + "/" + name, str(tmp),
                             Config=TransferConfig(max_concurrency=16))
        except Exception as e:
            tmp.unlink(missing_ok=True)
            fail(
                f"could not download {uri}{name}: {e}\n"
                "Fix: `bash scripts/download_artifacts.sh` (Hugging Face, gated model)."
            )
        tmp.rename(model_dir / name)


def size_cache(opt: dict, cache_dir: Path) -> None:
    """
    streaming only evicts once a split's cache reaches cache_limit, so the
    limit is what the cache *will* use: size it to the shard blocks in
    flight (current + prefetched next, with slack), not to the free disk.
    The limit applies to each split; the test split (~15 GB) is read in
    order and fits under it.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    opt["cache_dir"] = str(cache_dir)
    if opt.get("cache_limit"):
        return
    block = int(opt["datasets"]["train"].get("shard_block") or 1)
    # >= 4 shards: streaming's own minimum
    limit = max(4, round(1.5 * (2 * block + 4))) * SHARD_GB
    free_gb = shutil.disk_usage(cache_dir).free / 1e9
    if 2 * limit > free_gb - DISK_HEADROOM_GB:
        fail(
            f"only {free_gb:.0f} GB free at {cache_dir}; the S3 cache needs {2 * limit:.0f} GB "
            f"(train + test, shard_block {block}) plus {DISK_HEADROOM_GB} GB headroom.\n"
            "Fix: free space, pass --cache-dir on a bigger disk, or lower datasets.train.shard_block."
        )
    opt["cache_limit"] = f"{int(limit * 1000)}mb"
    log(f"S3 cache {cache_dir} (≤ {limit:.0f} GB per split, {free_gb:.0f} GB free)")


# ============================================================
# Config
# ============================================================

def _absolutize(value, base: Path):
    if not value or _is_s3(value):
        return value
    if isinstance(value, list):
        return [_absolutize(v, base) for v in value]
    p = Path(os.path.expanduser(str(value)))
    return str(p if p.is_absolute() else (base / p).resolve())


def build_config(cfg_path: Path, cache_dir: Path) -> tuple[dict, Path]:
    """
    The config with every relative path made absolute (the effective config
    is written elsewhere, so parse_yml's relative resolution must not apply),
    the cache filled in, and the run-specific assets checked.
    """
    opt = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    base = cfg_path.parent

    for split in ("train", "test"):
        ds = opt["datasets"][split]
        ds["dataroot"] = _absolutize(ds["dataroot"], base)

    bb = opt["structure"]["backbone"]
    bb["model_path"] = _absolutize(bb["model_path"], base)

    ts = opt["train_settings"]
    for key in ("save_checkpoint_folder_path", "load_checkpoint_file_path"):
        ts[key] = _absolutize(ts.get(key), base)
    pre = opt.get("pretraining_settings") or {}
    pre["checkpoint_path"] = _absolutize(pre.get("checkpoint_path"), base)
    ev = opt.get("eval_settings") or {}
    if ev.get("checkpoint_path"):
        ev["checkpoint_path"] = _absolutize(ev["checkpoint_path"], base)

    roots = [r for s in ("train", "test") for r in (lambda d: d if isinstance(d, list) else [d])(opt["datasets"][s]["dataroot"])]
    if any(_is_s3(r) for r in roots):
        check_aws()
        size_cache(opt, cache_dir)
    else:
        for r in roots:
            if not (Path(r) / "index.json").is_file():
                fail(f"no MDS index.json at {r}")

    ensure_backbone(Path(bb["model_path"]))

    fd, tmp = tempfile.mkstemp(prefix="local_config_", suffix=".yml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(opt, f, sort_keys=False)
    return opt, Path(tmp)


def _sagemaker_entry():
    """sagemaker/train.py as a module (not importable by name: the directory
    would shadow the `sagemaker` SDK package), for its resume logic."""
    spec = importlib.util.spec_from_file_location(
        "_sagemaker_entry", REPO / "sagemaker" / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=None,
                    help=f"training yml (default {DEFAULT_CONFIG.relative_to(REPO)})")
    ap.add_argument("--smoke", action="store_true",
                    help=f"use {SMOKE_CONFIG.relative_to(REPO)}: smoke slice, 1 epoch")
    ap.add_argument("--epochs", type=int, default=None,
                    help="stop after this epoch (default: epoch_settings.total_epochs)")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                    help=f"local cache for S3 shards (default {DEFAULT_CACHE_DIR})")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="run even without a usable GPU (very slow)")
    args = ap.parse_args()

    if args.smoke and args.config:
        ap.error("--smoke and --config are exclusive")
    cfg_path = (args.config or (SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG)).resolve()
    if not cfg_path.is_file():
        fail(f"config not found: {cfg_path}")

    sys.path.insert(0, str(REPO))
    os.environ.setdefault("TQDM_MININTERVAL", "30")

    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = "unknown"
    log(f"config {cfg_path.relative_to(REPO) if cfg_path.is_relative_to(REPO) else cfg_path} · git {sha or 'unknown'}")

    check_gpu(args.allow_cpu)
    opt, local_cfg = build_config(cfg_path, args.cache_dir.expanduser().resolve())

    end_epoch = args.epochs or int((opt.get("epoch_settings") or {}).get("total_epochs") or 1)

    # same restart detection as a SageMaker job: latest_checkpoint.pth -> resume
    effective = _sagemaker_entry()._effective_config(local_cfg)
    ckpt_root = Path(opt["train_settings"]["save_checkpoint_folder_path"]) / opt["name"]
    log(f"checkpoints {ckpt_root} · end_epoch {end_epoch}")

    try:
        from authgenforge.options.load import load_pipeline_from_yml

        _, _, _, trainer = load_pipeline_from_yml(str(effective))
        if trainer.epoch_start >= end_epoch:
            log(f"already trained to epoch {trainer.epoch_start} (≥ {end_epoch}) — nothing to do; "
                "raise --epochs to continue")
            return
        trainer.train_model(end_epoch=end_epoch)
    except KeyboardInterrupt:
        latest = ckpt_root / "latest_checkpoint.pth"
        if latest.is_file():
            log(f"interrupted — run the same command again to resume from {latest}")
        else:
            log("interrupted before the first checkpoint — a rerun starts over")
        sys.exit(130)
    finally:
        for p in {local_cfg, effective}:
            p.unlink(missing_ok=True)

    log(f"done — best checkpoint {ckpt_root / (opt['name'] + '_best.pth')}")


if __name__ == "__main__":
    main()
