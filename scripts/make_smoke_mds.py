#!/usr/bin/env python
"""
Build the tiny MDS dataset the SageMaker smoke test trains on.

Two sources, same output layout (<out>/train/, <out>/test/, each an MDS split
with index.json + shard.*.mds and the exact columns of processed-v1 — the
schema comes from packages/mdsconverter/build_mds_dataset.py's COLUMNS):

  real (default)  a class-balanced slice of processed-v1: the first shard(s)
                  of s3://authenta-data-rnd/image-tampering-detection/processed-v1/
                  {train,test} are downloaded and N samples copied byte-for-byte.
  --synthetic     generated images + rectangle masks, no AWS needed. Used by
                  tests/sagemaker/emulate_training_job.py and CI.

--upload syncs <out>/ to s3://.../image-tampering-detection/smoke/, which is
where infra/smoke-pipeline.asl.json mounts it from.

    python scripts/make_smoke_mds.py --out smoke_mds --upload
    python scripts/make_smoke_mds.py --synthetic --out /tmp/smoke_mds --train 8 --test 4
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "mdsconverter"))

from build_mds_dataset import COLUMNS, _writer_out_path  # noqa: E402

SOURCE = "s3://authenta-data-rnd/image-tampering-detection/processed-v1"
SMOKE = "s3://authenta-data-rnd/image-tampering-detection/smoke"


# ------------------------------------------------------------------
# Writing
# ------------------------------------------------------------------

def write_split(rows: list[dict], out_dir: Path) -> None:
    from streaming import MDSWriter

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    with _writer_out_path(str(out_dir)) as writer_out:
        with MDSWriter(out=writer_out, columns=COLUMNS, compression=None) as writer:
            for row in rows:
                writer.write(row)

    labels = [r["label"] for r in rows]
    print(f"[smoke] {out_dir}: {len(rows)} samples "
          f"({labels.count(0)} authentic / {labels.count(1)} tampered)")


# ------------------------------------------------------------------
# Synthetic source
# ------------------------------------------------------------------

def synthetic_rows(n: int, split: str, seed: int) -> list[dict]:
    """
    Random-texture JPEGs; tampered ones get a pasted rectangle and the
    matching PNG mask. Sizes vary and are not multiples of 16, so the
    crop/pad path in the transforms is exercised the way real data does.
    """
    import numpy as np
    from PIL import Image

    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    rows = []

    for i in range(n):
        tampered = i % 2 == 1
        w, h = rng.randint(560, 720), rng.randint(530, 650)
        img = nrng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        if tampered:
            x0, y0 = rng.randint(0, w // 2), rng.randint(0, h // 2)
            x1, y1 = x0 + rng.randint(64, w // 2), y0 + rng.randint(64, h // 2)
            img[y0:y1, x0:x1] = nrng.integers(0, 256, 3, dtype=np.uint8)
            mask[y0:y1, x0:x1] = 255

        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=90)
        image_bytes = buf.getvalue()

        mask_bytes = b""
        if tampered:
            mbuf = io.BytesIO()
            Image.fromarray(mask).save(mbuf, format="PNG")
            mask_bytes = mbuf.getvalue()

        label_str = "tampered" if tampered else "authentic"
        rows.append({
            "image": image_bytes,
            "mask": mask_bytes,
            "label": int(tampered),
            "label_str": label_str,
            "dataset": "synthetic",
            "split": split,
            "orig_path": f"synthetic/images/{label_str}/{split}_{i:05d}.jpg",
            "mask_orig_path": f"synthetic/masks/{split}_{i:05d}.png" if tampered else "",
            "ext": ".jpg",
            "mask_ext": ".png" if tampered else "",
            "width": w,
            "height": h,
            "filesize_bytes": len(image_bytes),
        })

    return rows


# ------------------------------------------------------------------
# Real source (a slice of processed-v1)
# ------------------------------------------------------------------

def _s3_cp(src: str, dst: Path) -> None:
    subprocess.run(["aws", "s3", "cp", src, str(dst), "--only-show-errors"], check=True)


def real_rows(split: str, n: int, cache: Path) -> list[dict]:
    """
    n samples from processed-v1/<split>, half authentic and half tampered,
    copied unchanged. Reads shards in order and stops as soon as both
    classes are filled, so usually only shard 0 (~512 MB) is downloaded.
    """
    from streaming import StreamingDataset

    split_cache = cache / split
    split_cache.mkdir(parents=True, exist_ok=True)

    index_path = split_cache / "full_index.json"
    if not index_path.exists():
        _s3_cp(f"{SOURCE}/{split}/index.json", index_path)
    index = json.loads(index_path.read_text())

    if set(index["shards"][0]["column_names"]) != set(COLUMNS):
        raise SystemExit(f"[smoke] processed-v1/{split} columns differ from build_mds_dataset.COLUMNS")

    want = {0: n // 2, 1: n - n // 2}
    picked: dict[int, list[dict]] = {0: [], 1: []}

    for k, shard in enumerate(index["shards"]):
        basename = shard["raw_data"]["basename"]
        if not (split_cache / basename).exists():
            print(f"[smoke] downloading {split}/{basename} ...")
            _s3_cp(f"{SOURCE}/{split}/{basename}", split_cache / basename)

        # A one-shard index next to the shard turns it into a valid
        # standalone MDS dir StreamingDataset can read in place.
        shard_dir = split_cache / f"shard{k}"
        shard_dir.mkdir(exist_ok=True)
        if not (shard_dir / basename).exists():
            os.replace(split_cache / basename, shard_dir / basename)
        (shard_dir / "index.json").write_text(json.dumps({"version": 2, "shards": [shard]}))

        ds = StreamingDataset(local=str(shard_dir), shuffle=False)
        for i in range(len(ds)):
            row = ds[i]
            label = int(row["label"])
            if len(picked[label]) < want[label]:
                picked[label].append({c: row[c] for c in COLUMNS})
            if all(len(picked[c]) >= want[c] for c in want):
                break
        del ds

        if all(len(picked[c]) >= want[c] for c in want):
            break
    else:
        raise SystemExit(f"[smoke] processed-v1/{split} has fewer than {want} samples per class")

    # interleave so a small batch already sees both classes
    return [r for pair in zip(picked[0], picked[1]) for r in pair] + picked[1][len(picked[0]):]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="smoke_mds", help="output root; <out>/train and <out>/test are written")
    ap.add_argument("--train", type=int, default=64, help="train samples")
    ap.add_argument("--test", type=int, default=32, help="test samples")
    ap.add_argument("--synthetic", action="store_true", help="generate samples instead of slicing processed-v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=None, help="where processed-v1 shards are downloaded (default: a temp dir)")
    ap.add_argument("--upload", action="store_true", help=f"sync <out>/ to {SMOKE}/")
    args = ap.parse_args()

    out = Path(args.out).resolve()

    if args.synthetic:
        write_split(synthetic_rows(args.train, "train", args.seed), out / "train")
        write_split(synthetic_rows(args.test, "test", args.seed + 1), out / "test")
    else:
        cache = Path(args.cache) if args.cache else Path(tempfile.mkdtemp(prefix="smoke_src_"))
        write_split(real_rows("train", args.train, cache), out / "train")
        write_split(real_rows("test", args.test, cache), out / "test")
        if not args.cache:
            shutil.rmtree(cache, ignore_errors=True)

    if args.upload:
        if args.synthetic:
            raise SystemExit("[smoke] refusing to upload synthetic data over the real smoke slice")
        # --delete: a smaller rebuild must not leave stale shards next to the
        # new index.json (they'd be ignored, but they cost a FastFile listing).
        subprocess.run(["aws", "s3", "sync", str(out), f"{SMOKE}/", "--delete", "--only-show-errors"], check=True)
        print(f"[smoke] uploaded -> {SMOKE}/")


if __name__ == "__main__":
    main()
