#!/usr/bin/env python3
"""
Independent post-hoc check for an MDS dataset built by build_mds_dataset.py.

For every sample:
  - re-derives authentic/tampered from the `authentic`/`tampered` path
    segment still visible in its stored `orig_path` and compares that
    against the stored `label`/`label_str` columns — deliberately not
    "trust label_str", so it catches a mismatch instead of re-reporting
    whatever the writer already believed
  - checks mask presence agrees with the label (tampered -> non-empty mask
    bytes + mask_orig_path, authentic -> neither)

Reports per-dataset authentic/tampered counts (compare against DATASETS.md).

Optional:
  --decode-check N                 fully decode N random samples' image + mask
                                   from the shard bytes and check sizes match
  --spot-check N --source-root DIR re-read N random samples' original image
                                   + mask off disk and compare bytes exactly —
                                   catches truncation/corruption anywhere in
                                   the read -> pool -> write path

Usage:
    python packages/mdsconverter/verify_mds_dataset.py --mds D:/forensics_mds/train
    python packages/mdsconverter/verify_mds_dataset.py --mds D:/forensics_mds/train \\
        --decode-check 200 --spot-check 200 --source-root D:/forensics_raw

Exit code 0 = clean, 1 = mismatches found (see stdout for details).
"""

from __future__ import annotations

import argparse
import io
import os
import random
import sys

from streaming import StreamingDataset

LABEL_TO_INT = {"authentic": 0, "tampered": 1}  # must match _forensics_common.LABEL_TO_INT


def infer_label_from_path(orig_path: str) -> str | None:
    parts = orig_path.replace("\\", "/").split("/")
    for p in parts[:-1]:
        if p.lower() in LABEL_TO_INT:
            return p.lower()
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mds", required=True, help="Path to one split's MDS dir, e.g. D:/forensics_mds/train")
    ap.add_argument("--source-root", default=None,
                     help="data_root the dataset was built from — required for --spot-check "
                          "(orig_path / mask_orig_path are relative to it).")
    ap.add_argument("--spot-check", type=int, default=0,
                     help="Randomly sample N rows and compare image+mask bytes to the originals on disk.")
    ap.add_argument("--decode-check", type=int, default=0,
                     help="Randomly sample N rows and fully decode image+mask from the shard bytes.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = StreamingDataset(local=args.mds, shuffle=False, batch_size=1)
    n = len(ds)
    print(f"{args.mds}: {n} samples")

    problems = []
    label_counts = {"authentic": 0, "tampered": 0}
    per_dataset: dict[str, dict[str, int]] = {}
    for i in range(n):
        s = ds[i]
        actual = s["label_str"]
        label_counts[actual] = label_counts.get(actual, 0) + 1
        per_dataset.setdefault(s["dataset"], {"authentic": 0, "tampered": 0})[actual] += 1

        expected = infer_label_from_path(s["orig_path"])
        if expected is None:
            problems.append((i, s["orig_path"], "orig_path has no authentic/tampered segment"))
        elif expected != actual:
            problems.append((i, s["orig_path"], f"path says {expected!r} but label_str={actual!r}"))
        if s["label"] != LABEL_TO_INT.get(actual):
            problems.append((i, s["orig_path"], f"label_str={actual!r} but label(int)={s['label']!r}"))

        has_mask = len(s["mask"]) > 0
        if actual == "tampered" and not (has_mask and s["mask_orig_path"]):
            problems.append((i, s["orig_path"], "tampered sample without mask bytes/mask_orig_path"))
        if actual == "authentic" and (has_mask or s["mask_orig_path"]):
            problems.append((i, s["orig_path"], "authentic sample carries a mask"))

    print(f"\nlabel counts: {label_counts}")
    print("per-dataset breakdown (authentic / tampered):")
    for name, c in sorted(per_dataset.items()):
        print(f"  {name}: {c['authentic']} / {c['tampered']}")

    rng = random.Random(args.seed)

    if args.decode_check:
        from PIL import Image
        idxs = rng.sample(range(n), min(args.decode_check, n))
        bad = 0
        for i in idxs:
            s = ds[i]
            try:
                im = Image.open(io.BytesIO(s["image"]))
                im.load()
                if (s["width"], s["height"]) != im.size:
                    raise ValueError(f"stored size {s['width']}x{s['height']} != decoded {im.size}")
                if len(s["mask"]):
                    m = Image.open(io.BytesIO(s["mask"]))
                    m.load()
                    if m.size != im.size:
                        raise ValueError(f"mask size {m.size} != image size {im.size}")
            except Exception as e:
                bad += 1
                problems.append((i, s["orig_path"], f"decode-check: {e}"))
        print(f"\ndecode-check: {len(idxs)} samples decoded from shard bytes, {bad} failures")

    if args.spot_check:
        if not args.source_root:
            print("\n--spot-check needs --source-root (the data_root the dataset was built from)",
                  file=sys.stderr)
            sys.exit(2)
        idxs = rng.sample(range(n), min(args.spot_check, n))
        checked = byte_mismatches = 0
        for i in idxs:
            s = ds[i]
            pairs = [(s["orig_path"], s["image"])]
            if s["mask_orig_path"]:
                pairs.append((s["mask_orig_path"], s["mask"]))
            for rel, stored in pairs:
                src_path = os.path.join(args.source_root, *rel.split("/"))
                if not os.path.exists(src_path):
                    print(f"WARNING: source file missing for spot-check idx={i}: {src_path}")
                    continue
                with open(src_path, "rb") as f:
                    src_bytes = f.read()
                checked += 1
                if src_bytes != stored:
                    byte_mismatches += 1
                    problems.append((i, rel, "byte mismatch vs original file on disk"))
        print(f"\nspot-check: {checked} files (images + masks) compared against originals, "
              f"{byte_mismatches} byte mismatches")

    if problems:
        print(f"\n{len(problems)} PROBLEMS FOUND:")
        for i, path, what in problems[:50]:
            print(f"  idx={i} {path}: {what}")
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more")
    else:
        print("\nclean: labels agree with orig_path, masks present exactly for tampered samples"
              f"{', all checked samples decode/match' if (args.decode_check or args.spot_check) else ''}.")

    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
