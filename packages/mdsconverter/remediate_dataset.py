#!/usr/bin/env python3
"""
Stage 2 of the forensics dataset pipeline — acts on a CSV report from
validate_dataset.py: deletes or quarantines (moves) every row with
status=failed, together with its paired mask (if it has one). Does not
re-scan or re-validate anything itself, just reads the report and acts on it.

Destructive by nature, so this is dry-run by default: without --execute, it
only prints what it would do.

action: delete    -> os.remove() the image and its mask
action: move      -> moves the image into <quarantine_dir>/<dataset>/<label>/<basename>
                      and its mask into <quarantine_dir>/<dataset>/mask/<basename>,
                      so a false positive can be moved back by hand

--reasons restricts which failure kinds are acted on (the part of the
report's `reason` before the first colon, e.g. image-decode-invalid). Worth
using: a missing-mask failure across a whole dataset usually means the mask
naming convention differs (fix with mask_suffixes:), not that every image
is bad — deleting those would throw away good data.

Usage:
    python packages/mdsconverter/remediate_dataset.py --config configs/mds/remediate_dataset.yml             # dry run
    python packages/mdsconverter/remediate_dataset.py --config configs/mds/remediate_dataset.yml --execute   # for real
    python packages/mdsconverter/remediate_dataset.py --report report.csv --action delete \\
        --reasons image-decode-invalid empty-mask
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time

from _forensics_common import cfg_picker, load_yaml_config, tqdm


def failure_kind(row: dict) -> str:
    return (row.get("reason") or "").split(":", 1)[0]


def load_failed_rows(report_path: str, reasons: set[str] | None) -> list[dict]:
    with open(report_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "failed"]
    if reasons:
        rows = [r for r in rows if failure_kind(r) in reasons]
    return rows


def targets(row: dict, quarantine_dir: str | None) -> list[tuple[str, str | None]]:
    """[(src, dest_or_None), ...] — the image, plus its mask when there is one."""
    ds = row.get("dataset") or "unknown"
    out = [(row["path"], os.path.join(quarantine_dir, ds, row.get("label") or "unknown",
                                      os.path.basename(row["path"])) if quarantine_dir else None)]
    if row.get("mask_path"):
        out.append((row["mask_path"], os.path.join(quarantine_dir, ds, "mask",
                                                   os.path.basename(row["mask_path"])) if quarantine_dir else None))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="Path to a yaml config (see configs/mds/remediate_dataset.yml).")
    ap.add_argument("--report", default=None, help="validate_dataset.py's CSV report.")
    ap.add_argument("--action", choices=["delete", "move"], default=None)
    ap.add_argument("--quarantine-dir", default=None, help="Required for --action move.")
    ap.add_argument("--reasons", nargs="+", default=None,
                     help="Only act on these failure kinds. Default: all failed rows.")
    ap.add_argument("--execute", action="store_true", default=None,
                     help="Actually perform the action. Without this: dry run (prints what would happen, "
                          "touches nothing).")
    cli = ap.parse_args()

    cfg = load_yaml_config(cli.config, path_keys=("report", "quarantine_dir")) if cli.config else {}
    pick = cfg_picker(cli, cfg)

    report = pick(cli.report, "report", None)
    if not report:
        ap.error("--report is required (either as a flag or `report:` in --config)")
    action = pick(cli.action, "action", None)
    if action not in ("delete", "move"):
        ap.error("--action delete|move is required (either as a flag or `action:` in --config)")
    quarantine_dir = pick(cli.quarantine_dir, "quarantine_dir", None)
    if action == "move" and not quarantine_dir:
        ap.error("--quarantine-dir is required when --action move")
    reasons = pick(cli.reasons, "reasons", None)
    reasons = set(reasons) if reasons else None
    execute = pick(cli.execute, "execute", False)

    rows = load_failed_rows(report, reasons)
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[failure_kind(r)] = kinds.get(failure_kind(r), 0) + 1
    print(f"{report}: {len(rows)} failed samples to {action}"
          f"{f' (restricted to reasons {sorted(reasons)})' if reasons else ''}")
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    if not rows:
        return

    q = quarantine_dir if action == "move" else None
    if not execute:
        print("\nDRY RUN — nothing will be touched. Preview:")
        for r in rows[:20]:
            for src, dest in targets(r, q):
                print(f"  {src}{f' -> {dest}' if dest else ' (delete)'}")
        if len(rows) > 20:
            print(f"  ... and {len(rows) - 20} more samples")
        print(f"\nRe-run with --execute to actually {action} these {len(rows)} samples (+ their masks).")
        return

    t0 = time.time()
    done = missing = errors = 0
    for r in tqdm(rows, desc="deleting" if action == "delete" else "moving", unit="sample"):
        for src, dest in targets(r, q):
            if not os.path.exists(src):
                missing += 1
                continue
            try:
                if action == "delete":
                    os.remove(src)
                else:
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.move(src, dest)
                done += 1
            except OSError as e:
                errors += 1
                print(f"WARNING: failed to {action} {src}: {e}", file=sys.stderr)

    dt = time.time() - t0
    print(f"\n{action}d {done} files ({len(rows)} samples incl. masks) in {dt:.1f}s "
          f"({missing} already gone, {errors} errors)")
    if action == "move":
        print(f"quarantined under {quarantine_dir}")


if __name__ == "__main__":
    main()
