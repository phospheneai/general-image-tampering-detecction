"""
Smoke test for the MDS data path, end to end on a tiny synthetic dataset:
  1. writes two fake datasets (JPEG / PNG / TIFF images, PNG masks, one
     "_mask"-suffixed, plus one truncated JPEG that must be skipped) into
     a temp dir
  2. converts them with packages/mdsconverter/build_mds_dataset.py and
     checks them with verify_mds_dataset.py (--spot-check, --decode-check)
  3. checks the stored image/mask bytes are byte-identical to the files
     (nothing re-encoded)
  4. loads the shards through ForensicsMDSDataset + a num_workers=2
     DataLoader and checks keys, shapes, dtypes, value ranges, and that
     tampered samples carry a non-empty mask

With --remote s3://.../<split> it also streams that split from S3 into a
fresh temp cache dir and loads a few batches in order (only the first shard
is downloaded). S3 credentials come from boto3's default chain, e.g.
AWS_PROFILE=<profile> when the machine's instance role can't read the bucket.

Run from the project root:
    ~/venv/bin/python tests/smoke_test_mds.py
    AWS_PROFILE=<profile> ~/venv/bin/python tests/smoke_test_mds.py --remote s3://authenta-data-rnd/image-tampering-detection/processed-v1/train
"""

import argparse
import io
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from streaming import StreamingDataset
from torch.utils.data import DataLoader

from authgenforge.augmentations.presets import get_train_transforms, get_val_transforms
from authgenforge.data.forensics_mds_dataset import ForensicsMDSDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONVERTER = os.path.join(ROOT, "packages", "mdsconverter")
CROP = 128


def _write_image(path, fmt, seed, size=(160, 120)):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path, format=fmt, **({"quality": 83} if fmt == "JPEG" else {}))


def _write_mask(path, size=(160, 120)):
    arr = np.zeros((size[1], size[0]), dtype=np.uint8)
    arr[30:80, 40:100] = 255
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path, format="PNG")


def make_raw(raw):
    # DS_A: authentic + tampered, masks by exact stem
    for i in range(6):
        _write_image(f"{raw}/DS_A/images/authentic/a{i}.jpg", "JPEG", i)
    for i, (ext, fmt) in enumerate([("jpg", "JPEG"), ("png", "PNG"), ("tif", "TIFF")] * 2):
        _write_image(f"{raw}/DS_A/images/tampered/t{i}.{ext}", fmt, 100 + i)
        _write_mask(f"{raw}/DS_A/masks/tampered/t{i}.png")
    # DS_B: tampered only, IMD2020-style "_mask" suffix
    for i in range(4):
        _write_image(f"{raw}/DS_B/images/tampered/b{i}.jpg", "JPEG", 200 + i)
        _write_mask(f"{raw}/DS_B/masks/tampered/b{i}_mask.png")
    # one truncated JPEG, like the ~0.17% of LAION-Mobile found in the raw
    # data — validation must skip it and log it to train_failed.csv
    _write_image(f"{raw}/DS_A/images/authentic/truncated.jpg", "JPEG", 999)
    with open(f"{raw}/DS_A/images/authentic/truncated.jpg", "r+b") as f:
        f.truncate(f.seek(0, 2) * 2 // 3)
    return 16  # valid samples


def run(cmd):
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"FAILED: {cmd[1]}")
    return r.stdout


def check_loader(ds, n_batches=2, batch_size=4, shuffle=True):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    for b, batch in enumerate(loader):
        assert set(batch) == {"image", "mask", "edge_mask"}, sorted(batch)
        bs = batch["image"].shape[0]
        assert batch["image"].shape == (bs, 3, CROP, CROP), batch["image"].shape
        assert batch["mask"].shape == (bs, 1, CROP, CROP), batch["mask"].shape
        assert batch["edge_mask"].shape == (bs, 1, CROP, CROP), batch["edge_mask"].shape
        for k in ("image", "mask", "edge_mask"):
            assert batch[k].dtype == torch.float32, (k, batch[k].dtype)
        assert 0.0 <= batch["mask"].min() and batch["mask"].max() <= 1.0
        if b == 0:
            for k, v in batch.items():
                print(f"    {k:10s} {tuple(v.shape)} {v.dtype} min={v.min():.3f} max={v.max():.3f}")
        if b + 1 >= n_batches:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default=None, help="s3://.../<split> to also test streaming from")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        raw, out = f"{tmp}/raw", f"{tmp}/mds"
        n = make_raw(raw)

        print("[1] convert")
        run([sys.executable, f"{CONVERTER}/build_mds_dataset.py", "--data-root", raw,
             "--datasets", "DS_A:train", "DS_B:train", "--mask-suffixes", "", "_mask",
             "--out", out, "--num-workers", "2", "--shard-size-mb", "1"])

        with open(f"{out}/train_failed.csv") as f:
            failed = f.read()
        assert "truncated.jpg" in failed and "image-decode-invalid" in failed, failed
        print("    truncated.jpg skipped and logged to train_failed.csv")

        print("[2] verify")
        print("   ", run([sys.executable, f"{CONVERTER}/verify_mds_dataset.py", "--mds", f"{out}/train",
                          "--decode-check", str(n), "--spot-check", str(n),
                          "--source-root", raw]).strip().splitlines()[-1])

        print("[3] bytes are the original files")
        sds = StreamingDataset(local=f"{out}/train", shuffle=False)
        assert len(sds) == n, (len(sds), n)
        for i in range(n):
            s = sds[i]
            with open(f"{raw}/{s['orig_path']}", "rb") as f:
                assert f.read() == s["image"], s["orig_path"]
            if s["label_str"] == "tampered":
                with open(f"{raw}/{s['mask_orig_path']}", "rb") as f:
                    assert f.read() == s["mask"], s["mask_orig_path"]
            else:
                assert s["mask"] == b"" and s["mask_orig_path"] == ""
        print(f"    {n}/{n} samples byte-identical")
        del sds

        print("[4] ForensicsMDSDataset, local, DataLoader(num_workers=2)")
        ds = ForensicsMDSDataset(f"{out}/train", transform=get_train_transforms(crop_size=CROP))
        assert len(ds) == n
        tampered_masks = [ds[i]["mask"].sum().item() > 0
                          for i in range(n) if ds.get_raw(i)["label_str"] == "tampered"]
        assert tampered_masks and all(tampered_masks), "tampered sample lost its mask"
        check_loader(ds)

    if args.remote:
        print(f"[5] ForensicsMDSDataset, streaming {args.remote}")
        with tempfile.TemporaryDirectory() as cache:
            ds = ForensicsMDSDataset(args.remote, transform=get_val_transforms(crop_size=CROP),
                                     cache_dir=cache)
            assert len(ds) > 0
            raw0 = ds.get_raw(0)
            print(f"    {len(ds):,} samples; sample 0: {raw0['dataset']} / {raw0['label_str']} / "
                  f"{raw0['width']}x{raw0['height']}")
            # in order, not shuffled: consecutive samples share a shard, so
            # this only downloads the first 512 MB shard instead of one
            # shard per random sample
            check_loader(ds, n_batches=3, batch_size=8, shuffle=False)

    print("\nPASS")


if __name__ == "__main__":
    main()
