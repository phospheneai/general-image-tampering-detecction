# MDS dataset conversion

`packages/mdsconverter/build_mds_dataset.py` combines the general image
forgery datasets (see [DATASETS.md](DATASETS.md)) from loose image/mask
folders into [MosaicML Streaming](https://github.com/mosaicml/streaming) (MDS)
shards — one MDS dataset per split, sharded, shuffled and
authentic:tampered-balanced — read at train time by
[`ForensicsMDSDataset`](authgenforge/data/forensics_mds_dataset.py).

This is stage 3 of the validate → remediate → convert pipeline; see
[DATASET_PIPELINE.md](DATASET_PIPELINE.md) for the DAG and `run_pipeline.py`.

## Quick start

```bash
python packages/mdsconverter/build_mds_dataset.py --config configs/mds/mds_dataset.yml --split train
python packages/mdsconverter/build_mds_dataset.py --config configs/mds/mds_dataset.yml --split test
```

Any CLI flag overrides the matching config key for that one run
(`--num-workers 3`, `--only-datasets MISD`, `--limit 500`, ...).

## Expected input structure

One folder per dataset. [`configs/mds/datasets.yml`](configs/mds/datasets.yml)
is the one file that says where each lives and which split it belongs to:

```
<folder>/<dataset>/
  images/authentic/   *.jpg|png|tif...
  images/tampered/    *.jpg|png|tif...
  masks/tampered/     one mask per tampered image (mask/ also accepted)
```

- Authentic images need no mask; an all-zero mask is built at load time.
- A tampered image `x.jpg` pairs with `masks/tampered/x<suffix>.<ext>`,
  trying each of `mask_suffixes` in order (`""` first, then `_mask`, ...).
- Subfolders under `authentic/`, `tampered/` and `masks/tampered/` are fine;
  folder names are matched case-insensitively.
- A dataset with only one class (COCO2017, LAION-Mobile, tampCOCO, DEFACTO)
  simply leaves the other folder out.
- `split` is never inferred from the path — always the value in `datasets.yml`.
- `dataset` is the folder's own name, so `COCO2017_test: {split: test, path: .../test/COCO2017}`
  is stored as `dataset=COCO2017, split=test`.

## Output layout

```
/home/ubuntu/data/processed/processed-v1/          (synced as-is to S3)
  train/  index.json  shard.00000.mds  shard.00001.mds ...
  test/   index.json  shard.00000.mds ...
  train_failed.csv  test_failed.csv                 (only if anything was skipped)

s3://authenta-data-rnd/image-tampering-detection/processed-v1/
  (same tree)
```

Train and test are fully independent MDS datasets (own `index.json`, own
shards). Shards are 512 MB (`shard_size_mb: 512`; the reference repo uses 256), uncompressed (JPEG/PNG are already compressed),
no hashes.

## Column schema

| column | type | meaning |
|---|---|---|
| `image` | bytes | original file bytes, **never re-encoded** (keeps JPEG compression traces intact) |
| `mask` | bytes | original mask file bytes; `b""` for authentic images |
| `label` | int | `0`=authentic, `1`=tampered |
| `label_str` | str | `"authentic"` / `"tampered"` |
| `dataset` | str | dataset folder name |
| `split` | str | from `datasets.yml` (`train`/`test`) |
| `orig_path` | str | `<dataset>/images/<label>/<file>`, `/`-separated, for traceability |
| `mask_orig_path` | str | `<dataset>/masks/tampered/<file>`; `""` for authentic |
| `ext`, `mask_ext` | str | lowercased extensions |
| `width`, `height` | int | image size |
| `filesize_bytes` | int | image file size |

## When a file fails

The converter never writes a tampered image without its mask. With
`validate: true` it also fully decodes every image + mask with PIL first.
Anything skipped gets one row in `<out>/<split>_failed.csv` (`path,reason`).
`mds_dataset.yml` sets `validate: false` because the pipeline's validate +
remediate stages already did that work — flip it on if you run this
standalone.

## Balance + shuffle

Within each split, authentic and tampered samples are each shuffled
independently (seed 42 and 43) and then interleaved proportionally
(Bresenham-style), so every contiguous window of the write order — and
therefore every physical shard — holds the split's global
authentic:tampered ratio, with datasets mixed inside each class.

## Speed and memory

Reading bytes (+ optional validation) runs in a `multiprocessing.Pool` of
`num_workers`; results come back in write order (`Pool.imap`) to the single
`MDSWriter`. `chunksize: 16` (images are small). Each discovered image is a
slotted `Sample` dataclass (~2.7M of them held in memory during the shuffle).

## Checking it

```bash
python packages/mdsconverter/verify_mds_dataset.py --mds /home/ubuntu/data/processed/processed-v1/train \
    --decode-check 500 --spot-check 500 \
    --source-root /home/ubuntu/data/raw/train /home/ubuntu/data/extracted
python packages/mdsconverter/verify_mds_dataset.py --mds /home/ubuntu/data/processed/processed-v1/test \
    --decode-check 500 --spot-check 500 --source-root /home/ubuntu/data/raw/test
```

Re-derives every label from `orig_path`, checks masks are present exactly
for tampered samples, prints per-dataset counts, decodes N samples and
compares N samples' bytes against the originals. Exit code 1 on any problem.

## Reading it back

In PyTorch, through `ForensicsMDSDataset` — returns
`{"image": (3,H,W) float, "mask": (1,H,W) float, "edge_mask": (1,H,W) float, "label": int64 (0 authentic / 1 tampered)}`:

```python
from torch.utils.data import DataLoader
from authgenforge.augmentations.presets import get_train_transforms
from authgenforge.data.forensics_mds_dataset import ForensicsMDSDataset

# local shards
ds = ForensicsMDSDataset("/home/ubuntu/data/processed/processed-v1/train",
                         transform=get_train_transforms(crop_size=512))

# or from S3, mounted as a folder first (needs mount-s3: scripts/install_mountpoint.sh,
# and s3:ListBucket + s3:GetObject; credentials from the usual AWS chain):
#   python -m authgenforge.utils.s3_mount \
#       s3://authenta-data-rnd/image-tampering-detection/ ~/data/s3/image-tampering-detection
ds = ForensicsMDSDataset("/home/ubuntu/data/s3/image-tampering-detection/processed-v1/train",
                         transform=get_train_transforms(crop_size=512))

loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
```

It is map-style (a plain `DataLoader` shards indices across workers with no
duplicates), and multiple roots can be given as a list. A random-order epoch
over an S3 root downloads every shard once, so the cache ends up the size of
the split. `num_workers > 0` needs Linux (fork).

For raw bytes and metadata without decoding: `ds.get_raw(i)`, or
`streaming.StreamingDataset(local=<split dir>)[i]`.

## Training on it

On any machine with the repo + `bash scripts/install_deps.sh` +
`bash scripts/download_artifacts.sh` (model weights), point the training
config at the processed dataset — local shards, or the S3 copy mounted as a
folder (no manual dataset download):

```bash
python tests/smoke_test_pipeline.py --config configs/train_forensics_mds.yml   # a few real steps
python tests/overfit_single_batch.py --config configs/train_forensics_mds.yml
```


[`configs/train_forensics_mds.yml`](configs/train_forensics_mds.yml):

```yaml
data_format: mds                    # -> _DATASET_BACKENDS["mds"] in authgenforge/options/load.py
datasets:
  train:
    dataroot: [/home/ubuntu/data/processed/processed-v1/train]
  test:
    dataroot: [/home/ubuntu/data/processed/processed-v1/test]
```
