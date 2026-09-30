#!/usr/bin/env python3
"""
Stage 0 (prepare) for Columbia — turns its edgemask ground truth into the
binary masks/tampered/<stem>.png the pipeline expects, writing a complete
dataset folder outside raw/ (raw/ is never modified):

    <src>/images/{authentic,tampered}/*.tif          copied byte-for-byte
    <src>/raw masks/tampered/<stem>_edgemask.jpg      -> <dst>/masks/tampered/<stem>.png

Columbia's edgemasks label regions by source camera, in 4 colours:
    bright red   (255,0,0)  camera 1, near the suspicious splicing boundary
    bright green (0,255,0)  camera 2, near the boundary
    regular red  (200,0,0)  camera 1, far from the boundary
    regular green(0,200,0)  camera 2, far from the boundary
The binary mask marks bright red only (camera 1, near the boundary) = 255,
everything else = 0. The edgemasks are lossy JPEGs, so each pixel takes the
nearest of the 4 palette colours (the few off-palette pixels are red/green
blends along region edges). <stem>_edgemask_3.jpg (a variant with an extra
blue band) is not used.

Idempotent: files already present with the right size are left alone.

    python packages/mdsconverter/prepare_columbia.py \\
        --src /home/ubuntu/data/raw/test/columbia --dst /home/ubuntu/data/extracted/columbia
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
from PIL import Image

PALETTE = np.array([
    [255, 0, 0],   # 0: bright red   -> tampered
    [0, 255, 0],   # 1: bright green
    [200, 0, 0],   # 2: regular red
    [0, 200, 0],   # 3: regular green
], dtype=np.int32)
TAMPERED_CLASSES = (0,)


def edgemask_to_binary(path: str) -> Image.Image:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int32)
    dist = ((rgb[..., None, :] - PALETTE) ** 2).sum(-1)
    nearest = dist.argmin(-1)
    return Image.fromarray(np.isin(nearest, TAMPERED_CLASSES).astype(np.uint8) * 255, mode="L")


def copy_if_needed(src: str, dst: str) -> bool:
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/home/ubuntu/data/raw/test/columbia")
    ap.add_argument("--dst", default="/home/ubuntu/data/extracted/columbia")
    ap.add_argument("--edgemask-dir", default="raw masks/tampered")
    ap.add_argument("--edgemask-suffix", default="_edgemask")
    args = ap.parse_args()

    if os.path.realpath(args.dst).startswith(os.path.realpath(args.src)):
        sys.exit("--dst must be outside --src")

    copied = 0
    for label in ("authentic", "tampered"):
        d = os.path.join(args.src, "images", label)
        for fn in sorted(os.listdir(d)):
            copied += copy_if_needed(os.path.join(d, fn), os.path.join(args.dst, "images", label, fn))

    mask_dir = os.path.join(args.dst, "masks", "tampered")
    os.makedirs(mask_dir, exist_ok=True)
    written, fractions, problems = 0, [], []
    for fn in sorted(os.listdir(os.path.join(args.src, "images", "tampered"))):
        stem = os.path.splitext(fn)[0]
        edge = os.path.join(args.src, args.edgemask_dir, f"{stem}{args.edgemask_suffix}.jpg")
        if not os.path.exists(edge):
            problems.append(f"{fn}: no {os.path.basename(edge)}")
            continue
        mask = edgemask_to_binary(edge)
        with Image.open(os.path.join(args.src, "images", "tampered", fn)) as im:
            if im.size != mask.size:
                problems.append(f"{fn}: image {im.size} vs mask {mask.size}")
                continue
        arr = np.asarray(mask)
        fractions.append((arr > 0).mean())
        if not arr.any():
            problems.append(f"{fn}: no bright-red pixels")
        mask.save(os.path.join(mask_dir, f"{stem}.png"))
        written += 1

    fr = np.array(fractions) if fractions else np.zeros(1)
    print(f"images copied: {copied} (existing ones kept)  masks written: {written}")
    print(f"tampered area per mask: min {fr.min():.4f}  median {np.median(fr):.4f}  max {fr.max():.4f}")
    for p in problems:
        print(f"PROBLEM: {p}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
