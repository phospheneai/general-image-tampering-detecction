#!/usr/bin/env python3
"""
Stage 1 of the forensics dataset pipeline — scans every configured dataset
folder and checks every sample, without touching anything on disk. Writes one
row per image to a CSV report (status=ok/failed + reason, plus
dataset/split/label/mask context) so a human can review exactly what failed
and why before anything gets deleted or moved.

What counts as a failure (see _forensics_common.validate_sample):
    unreadable / image-decode-invalid   image file can't be read or fully decoded
    missing-mask / ambiguous-mask       tampered image with no (or >1) matching mask
    mask-decode-invalid                 mask file can't be decoded
    mask-size-mismatch                  mask and image dimensions differ
    empty-mask                          tampered image whose mask marks no pixels

Same discovery + mask-pairing rules as build_mds_dataset.py (shared via
_forensics_common.py) — a sample this script flags is exactly the sample
build_mds_dataset.py would otherwise have written with --no-validate.

Usage:
    python packages/mdsconverter/validate_dataset.py --config configs/mds/validate_dataset.yml
    python packages/mdsconverter/validate_dataset.py --config configs/mds/validate_dataset.yml --split train

    # plain CLI, no yaml
    python packages/mdsconverter/validate_dataset.py --data-root D:/forensics_raw \\
        --datasets CASIAv2:train Columbia:test --report-out D:/forensics_raw/_logs/validation_report.csv

Requires: pip install pillow numpy tqdm pyyaml
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass

from _forensics_common import (
    Sample, add_dataset_args, cfg_picker, discover, load_yaml_config,
    resolve_dataset_args, tqdm, validate_sample,
)

_CHECK_MASK_CONTENT = True


def _init_worker(check_mask_content: bool):
    global _CHECK_MASK_CONTENT
    _CHECK_MASK_CONTENT = check_mask_content


@dataclass(slots=True)
class ValidationResult:
    sample: Sample
    ok: bool
    reason: str
    filesize_bytes: int
    width: int
    height: int


def check_one(sample: Sample) -> ValidationResult:
    ok, reason, filesize, w, h = validate_sample(sample, _CHECK_MASK_CONTENT)
    return ValidationResult(sample=sample, ok=ok, reason=reason, filesize_bytes=filesize, width=w, height=h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_args(ap)
    ap.add_argument("--report-out", default=None, help="Where to write the CSV report.")
    ap.add_argument("--num-workers", type=int, default=None, help="Default: os.cpu_count()")
    ap.add_argument("--chunksize", type=int, default=None, help="Default: 16")
    ap.add_argument("--no-check-mask-content", dest="check_mask_content", action="store_false", default=None,
                     help="Skip the empty-mask check (only verifies masks decode and match image size).")
    cli = ap.parse_args()

    cfg = load_yaml_config(cli.config, path_keys=("report_out", "data_root")) if cli.config else {}
    pick = cfg_picker(cli, cfg)

    report_out = pick(cli.report_out, "report_out", None)
    if not report_out:
        ap.error("--report-out is required (either as a flag or `report_out:` in --config)")
    specs, image_exts, mask_exts, mask_suffixes = resolve_dataset_args(ap, cli, cfg)
    num_workers = pick(cli.num_workers, "num_workers", os.cpu_count())
    chunksize = pick(cli.chunksize, "chunksize", 16)
    check_mask_content = pick(cli.check_mask_content, "check_mask_content", True)

    print(f"discovering {len(specs)} dataset(s), mask suffixes={mask_suffixes} ...")
    samples, _stats = discover(specs, image_exts, mask_exts, mask_suffixes)
    print(f"discovered {len(samples)} total images\n")
    if not samples:
        print("nothing found — check data_root / datasets: / folder layout", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(report_out)) or ".", exist_ok=True)
    t0 = time.time()
    n_ok = n_failed = 0
    per_dataset_failed: dict[str, dict[str, int]] = {}

    with open(report_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "mask_path", "dataset", "split", "label", "filesize_bytes",
                          "width", "height", "status", "reason"])
        with mp.Pool(num_workers, initializer=_init_worker, initargs=(check_mask_content,)) as pool:
            for r in tqdm(pool.imap(check_one, samples, chunksize=chunksize),
                          total=len(samples), desc="validating", unit="img"):
                s = r.sample
                status = "ok" if r.ok else "failed"
                writer.writerow([s.path, s.mask_path, s.dataset, s.split, s.label_str, r.filesize_bytes,
                                  r.width, r.height, status, r.reason])
                if r.ok:
                    n_ok += 1
                else:
                    n_failed += 1
                    kind = r.reason.split(":", 1)[0]
                    by_kind = per_dataset_failed.setdefault(s.dataset, {})
                    by_kind[kind] = by_kind.get(kind, 0) + 1

    dt = time.time() - t0
    print(f"\nvalidated {len(samples)} images in {dt:.1f}s ({len(samples) / dt if dt > 0 else 0:.1f} img/s)")
    print(f"  ok: {n_ok}")
    print(f"  failed: {n_failed}")
    if per_dataset_failed:
        print("  failed by dataset:")
        for name, kinds in sorted(per_dataset_failed.items(), key=lambda kv: -sum(kv[1].values())):
            detail = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
            print(f"    {name}: {sum(kinds.values())}  ({detail})")
    print(f"\nreport written to {report_out}")
    if n_failed:
        print(f"next: review {report_out}, then either fix/delete the failed files yourself or run "
              f"remediate_dataset.py --report {report_out} (dry-run by default)")


if __name__ == "__main__":
    main()
