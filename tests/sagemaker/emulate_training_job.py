#!/usr/bin/env python
"""
Run sagemaker/train.py the way a SageMaker training job does, without AWS,
and check the result against what the pipeline relies on.

A SageMaker job is: input channels mounted at /opt/ml/input/data/<channel>/,
/opt/ml/checkpoints synced to S3, a traceback in /opt/ml/output/failure on
error, and metrics scraped from the log with the regexes in the ASL. This
script recreates that layout under --root and verifies each part.

    prepare  synthetic MDS train/test (or --data, e.g. the real smoke slice) + a
             random-init DINOv3 ViT-L/16 backbone (same config.json, so the
             real model code path runs) under --root
    local    run train.py in this Python env with the smoke config's /opt/ml
             paths remapped to --root, then `check` — twice, the second run
             resuming from the first run's checkpoint like a spot restart
    check    verify a finished run (stdlib only, so it also runs on the host
             after a container run — see container_test.sh)

    python tests/sagemaker/emulate_training_job.py prepare --root D:/tmp/smjob
    python tests/sagemaker/emulate_training_job.py local   --root D:/tmp/smjob --crop-size 224
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = REPO_ROOT / "sagemaker" / "config" / "smoke.yml"
SMOKE_ASL = REPO_ROOT / "infra" / "smoke-pipeline.asl.json"
BACKBONE_CONFIG = REPO_ROOT / "sagemaker" / "backbone" / "dinov3-vitl16" / "config.json"


# ------------------------------------------------------------------
# prepare
# ------------------------------------------------------------------

def prepare(root: Path, n_train: int, n_test: int, data_from: Path | None) -> None:
    data = root / "input" / "data"
    if data_from:
        # e.g. the real smoke slice from scripts/make_smoke_mds.py
        for split in ("train", "test"):
            shutil.rmtree(data / split, ignore_errors=True)
            shutil.copytree(data_from / split, data / split)
            print(f"[prepare] {data_from / split} -> {data / split}")
    else:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from make_smoke_mds import synthetic_rows, write_split

        write_split(synthetic_rows(n_train, "train", seed=0), data / "train")
        write_split(synthetic_rows(n_test, "test", seed=1), data / "test")

    artifacts = data / "artifacts"
    if (artifacts / "model.safetensors").exists():
        print(f"[prepare] backbone already at {artifacts}")
    else:
        # Random weights from the real config: everything downstream
        # (AutoModel.from_pretrained, the ViT-L shape checks, LoRA wrapping)
        # runs exactly as with the gated pretrained weights.
        import torch
        from transformers import AutoConfig, AutoModel

        torch.manual_seed(0)
        config = AutoConfig.from_pretrained(BACKBONE_CONFIG.parent, local_files_only=True)
        AutoModel.from_config(config).save_pretrained(artifacts, safe_serialization=True)
        print(f"[prepare] random-init backbone -> {artifacts}")

    for d in ("checkpoints", "output"):
        (root / d).mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# local
# ------------------------------------------------------------------

def _local_config(root: Path, crop_size: int | None) -> Path:
    """The smoke config with /opt/ml remapped to root. Nothing else changes
    except crop size (optional: CPU speed) and worker count on Windows, where
    StreamingDataset can't be shared with spawned workers."""
    import yaml

    text = SMOKE_CONFIG.read_text(encoding="utf-8").replace("/opt/ml", root.as_posix())
    cfg = yaml.safe_load(text)
    if crop_size:
        cfg["datasets"]["train"]["crop_size"] = crop_size
        cfg["datasets"]["test"]["crop_size"] = crop_size
    if os.name == "nt":
        cfg["datasets"]["train"]["n_workers"] = 0
        cfg["datasets"]["test"]["n_workers"] = 0
        cfg["datasets"]["train"]["pin_memory"] = False
        cfg["datasets"]["test"]["pin_memory"] = False

    path = root / "smoke_local.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _run_train(config: Path, end_epoch: int, log: Path) -> int:
    cmd = [sys.executable, str(REPO_ROOT / "sagemaker" / "train.py"),
           "--config", str(config), "--end-epoch", str(end_epoch),
           "train"]  # the legacy positional SageMaker can append — must be ignored
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    print(f"[local] {' '.join(cmd)}", flush=True)
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=env,
                                cwd=str(Path.home()))  # not the repo: paths must not depend on cwd
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        return proc.wait()


def local(root: Path, crop_size: int | None) -> None:
    # a fresh job: a leftover checkpoint would make run 1 a resume
    for d in ("checkpoints", "output"):
        shutil.rmtree(root / d, ignore_errors=True)
        (root / d).mkdir(parents=True)
    config = _local_config(root, crop_size)

    first = root / "output" / "run1.log"
    rc = _run_train(config, 1, first)
    check(root, first, rc, expect_resume=False, end_epoch=1)

    # second launch on the same checkpoint dir = what SageMaker does after a
    # spot interruption: it restores /opt/ml/checkpoints and reruns the job
    second = root / "output" / "run2.log"
    rc = _run_train(config, 2, second)
    check(root, second, rc, expect_resume=True, end_epoch=2)


# ------------------------------------------------------------------
# check
# ------------------------------------------------------------------

def _csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check(root: Path, log: Path, returncode: int, expect_resume: bool, end_epoch: int) -> None:
    text = log.read_text(encoding="utf-8", errors="replace")
    problems = []

    if returncode != 0:
        problems.append(f"train.py exited with {returncode}")
    if "Traceback (most recent call last)" in text:
        problems.append("traceback in the log")
    if (root / "output" / "failure").exists():
        problems.append("/opt/ml/output/failure was written")

    # every metric the state machine scrapes must actually appear in the log,
    # otherwise the SageMaker Metrics tab silently stays empty
    asl = json.loads(SMOKE_ASL.read_text(encoding="utf-8"))
    metric_defs = asl["States"]["SmokeTrain"]["Parameters"]["AlgorithmSpecification"]["MetricDefinitions"]
    for m in metric_defs:
        values = re.findall(m["Regex"], text)
        if not values:
            problems.append(f"metric {m['Name']} never matched /{m['Regex']}/")
        else:
            print(f"[check] metric {m['Name']:<14} = {values[-1]}")

    if expect_resume and "[train] checkpoint resume detected" not in text:
        problems.append("restart did not pick up latest_checkpoint.pth")
    if not re.search(rf"\[Train\] Epoch {end_epoch} \|", text):
        problems.append(f"epoch {end_epoch} did not train")

    exp = root / "checkpoints" / "sm_smoke"
    for name in ["latest_checkpoint.pth", f"epoch{end_epoch}.pth"]:
        path = exp / name
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"missing checkpoint {path}")
        else:
            print(f"[check] {path.relative_to(root)}  {path.stat().st_size / 2**20:.0f} MiB")

    # result files: one metrics row per epoch, one prediction row per test image
    metrics_csv = exp / "metrics.csv"
    rows = _csv_rows(metrics_csv)
    if [r.get("epoch") for r in rows] != [str(e) for e in range(1, end_epoch + 1)]:
        problems.append(f"{metrics_csv} should have one row for each of epochs 1..{end_epoch}")
    else:
        print(f"[check] {metrics_csv.relative_to(root)}  {len(rows)} epoch row(s)")

    wanted = ["image_name", "ground_truth", "probability", "predicted_class", "iou"]
    found = sorted(exp.glob(f"*/predictions/val_epoch_{end_epoch}.csv"))
    rows = _csv_rows(found[-1]) if found else []
    if not rows or list(rows[0]) != wanted:
        problems.append(f"no per-image val_epoch_{end_epoch}.csv with columns {wanted}")
    else:
        print(f"[check] {found[-1].relative_to(root)}  {len(rows)} image row(s)")

    if problems:
        print("\n[check] FAILED:\n  - " + "\n  - ".join(problems))
        sys.exit(1)
    print(f"[check] OK: {log.name}\n")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--train", type=int, default=8)
    p.add_argument("--test", type=int, default=4)
    p.add_argument("--data", type=Path, default=None,
                   help="use <data>/train and <data>/test (MDS) instead of synthetic samples")

    p = sub.add_parser("local")
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--crop-size", type=int, default=None,
                   help="override the smoke config's crop (e.g. 224 for a CPU run)")

    p = sub.add_parser("check")
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--log", required=True, type=Path)
    p.add_argument("--returncode", type=int, default=0)
    p.add_argument("--expect-resume", action="store_true")
    p.add_argument("--end-epoch", type=int, default=1)

    a = ap.parse_args()
    # tqdm bars are echoed through; a Windows console codepage can't encode them
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = a.root.resolve()
    if a.cmd == "prepare":
        prepare(root, a.train, a.test, a.data)
    elif a.cmd == "local":
        local(root, a.crop_size)
    else:
        check(root, a.log, a.returncode, a.expect_resume, a.end_epoch)


if __name__ == "__main__":
    main()
