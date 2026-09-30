"""
Smoke test: load the real processed-v1 dataset straight from S3 through
ForensicsMDSDataset + a multi-worker DataLoader, exactly as training does.

For each split (train, test):
  1. opens s3://.../processed-v1/<split> with a fresh, empty cache dir
  2. checks it is a torch Dataset with the expected number of samples
  3. reads sample metadata (dataset, label, original path, original size)
  4. checks a tampered sample comes back with a non-empty mask and an
     authentic one with an all-zero mask
  5. loads batches with DataLoader(num_workers=4) and checks keys, shapes,
     dtypes and value ranges

Samples are read in order, so each split only downloads its first shard
(512 MB) plus index.json. The cache dir is deleted at the end.

S3 credentials come from boto3's default chain — set AWS_PROFILE when the
machine's instance role can't read the bucket. Run from the project root:
    AWS_PROFILE=<profile> python tests/smoke_test_s3.py
    AWS_PROFILE=<profile> python tests/smoke_test_s3.py --splits test --crop-size 512
"""

import argparse
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Dataset

from authgenforge.augmentations.presets import get_train_transforms, get_val_transforms
from authgenforge.data.forensics_mds_dataset import ForensicsMDSDataset

S3_ROOT = "s3://authenta-data-rnd/image-tampering-detection/processed-v1"
EXPECTED = {"train": 1_827_437, "test": 27_604}  # processed-v1 sample counts (DATASETS.md)


def check_split(split, crop, cache_dir, n_batches, batch_size):
    url = f"{S3_ROOT}/{split}"
    tf = get_train_transforms(crop_size=crop) if split == "train" else get_val_transforms(crop_size=crop)
    t0 = time.time()

    ds = ForensicsMDSDataset(url, transform=tf, cache_dir=cache_dir)
    assert isinstance(ds, Dataset), type(ds)
    assert len(ds) == EXPECTED[split], f"{split}: {len(ds)} samples, expected {EXPECTED[split]}"
    print(f"  torch Dataset: yes | samples: {len(ds):,} (expected {EXPECTED[split]:,})")

    raw = ds.get_raw(0)
    print(f"  sample 0: {raw['dataset']} / {raw['label_str']} / {raw['orig_path']} "
          f"({raw['width']}x{raw['height']} original)")

    # first shard only: find one sample of each label among the first rows
    seen = {}
    for i in range(min(200, len(ds))):
        label = ds.get_raw(i)["label_str"]
        if label not in seen:
            seen[label] = i
        if len(seen) == 2:
            break
    for label, i in sorted(seen.items()):
        marked = ds[i]["mask"].sum().item() > 0
        assert marked == (label == "tampered"), f"{label} sample {i}: mask marked={marked}"
        print(f"  {label:9s} sample {i}: mask {'has tampered pixels' if marked else 'all zero'} ✓")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    for b, batch in enumerate(loader):
        assert set(batch) == {"image", "mask", "edge_mask"}, sorted(batch)
        bs = batch["image"].shape[0]
        assert batch["image"].shape == (bs, 3, crop, crop), batch["image"].shape
        assert batch["mask"].shape == (bs, 1, crop, crop), batch["mask"].shape
        assert batch["edge_mask"].shape == (bs, 1, crop, crop), batch["edge_mask"].shape
        for k, v in batch.items():
            assert v.dtype == torch.float32, (k, v.dtype)
            assert torch.isfinite(v).all() and 0.0 <= v.min() and v.max() <= 1.0, k
        print(f"  batch {b}: " + ", ".join(f"{k}={tuple(v.shape)}" for k, v in batch.items()))
        if b + 1 >= n_batches:
            break

    print(f"  OK in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "test"], choices=sorted(EXPECTED))
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cache-dir", default=None,
                    help="Where shards are cached. Default: a fresh temp dir, deleted afterwards.")
    args = ap.parse_args()

    cache = args.cache_dir or tempfile.mkdtemp(prefix="mds_s3_smoke_")
    try:
        for split in args.splits:
            print(f"\n[{split}] streaming {S3_ROOT}/{split}  (cache: {cache})")
            check_split(split, args.crop_size, cache, args.batches, args.batch_size)
        downloaded = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(cache) for f in fs)
        print(f"\ndownloaded from S3: {downloaded / 2**20:.0f} MiB")
    finally:
        if not args.cache_dir:
            shutil.rmtree(cache, ignore_errors=True)

    print("\nPASS")


if __name__ == "__main__":
    main()
