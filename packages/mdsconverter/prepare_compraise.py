#!/usr/bin/env python3
"""
Stage 0 (prepare) for compRAISE — the raw copy only holds zips
(compRAISE_full.zip, plus compraise.zip / parts/ which are the same data
split into 15 volumes), so this extracts compRAISE_full.zip's JPEGs into
a normal dataset folder outside raw/ (raw/ is never modified):

    <zip>:compRAISE/<file>.jpg  ->  <dst>/images/authentic/<file>.jpg

zipfile verifies every member's CRC32 while reading, so a corrupt member
fails loudly instead of landing on disk. Idempotent: members already
extracted with the right size are skipped, so an interrupted run resumes.

    python packages/mdsconverter/prepare_compraise.py \\
        --zip /home/ubuntu/data/raw/train/compRAISE/compRAISE_full.zip \\
        --dst /home/ubuntu/data/extracted/compRAISE
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default="/home/ubuntu/data/raw/train/compRAISE/compRAISE_full.zip")
    ap.add_argument("--dst", default="/home/ubuntu/data/extracted/compRAISE")
    args = ap.parse_args()

    out_dir = os.path.join(args.dst, "images", "authentic")
    os.makedirs(out_dir, exist_ok=True)

    zf = zipfile.ZipFile(args.zip)
    members = [i for i in zf.infolist() if not i.is_dir()]
    done = skipped = bad = 0
    for i, info in enumerate(members):
        out = os.path.join(out_dir, os.path.basename(info.filename))
        if os.path.exists(out) and os.path.getsize(out) == info.file_size:
            skipped += 1
            continue
        try:
            with zf.open(info) as r, open(out + ".part", "wb") as w:
                shutil.copyfileobj(r, w, 1 << 22)
            os.replace(out + ".part", out)
            done += 1
        except Exception as e:
            bad += 1
            print(f"FAILED {info.filename}: {e}", flush=True)
        if i % 1000 == 0:
            print(f"{i}/{len(members)} processed, {bad} failed", flush=True)

    print(f"DONE: {done} extracted, {skipped} already present, {bad} failed "
          f"(of {len(members)}) -> {out_dir}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
