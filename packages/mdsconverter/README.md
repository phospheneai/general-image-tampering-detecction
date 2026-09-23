# mdsconverter — forensics datasets → MDS

Converts the general image forgery datasets (see [DATASETS.md](../../DATASETS.md))
from loose image/mask folders into MosaicML Streaming (MDS) shards, one MDS
dataset per split, read at train time by
[`ForensicsMDSDataset`](../../authgenforge/data/forensics_mds_dataset.py).

## Input layout

One folder per dataset under a common `data_root`:

```
<data_root>/
  CASIAv2/
    images/authentic/   *.jpg|png|tif...
    images/tampered/    *.jpg|png|tif...
    mask/tampered/      one mask per tampered image (masks/ also accepted)
  Columbia/
    ...
```

- Authentic images need no mask; an all-zero mask is built at load time.
- A tampered image `x.jpg` pairs with `mask/tampered/x<suffix>.<ext>`, trying
  each of `mask_suffixes` in order (`""` first, then `_gt`, `_mask`, ...).
- Subfolders under `authentic/`, `tampered/` and `mask/tampered/` are fine.
- Folder names are matched case-insensitively.
- A dataset with only one class (e.g. LAION-Mobile, DEFACTO) simply leaves the
  other folder out.

## Pipeline

| Stage | Script | Config | What it does |
|---|---|---|---|
| — | — | `configs/mds/datasets.yml` | **The one file to edit**: `data_root`, dataset → split map, mask suffixes |
| 1 validate | `validate_dataset.py` | `validate_dataset.yml` | Decodes every image + mask and writes a CSV report. Touches nothing on disk |
| 2 remediate | `remediate_dataset.py` | `remediate_dataset.yml` | Moves or deletes the failed samples (and their masks). Dry run unless `--execute` |
| 3 convert | `build_mds_dataset.py` | `mds_dataset.yml` | Writes `<out>/<split>/` shards, shuffled and balanced authentic:tampered per shard |
| check | `verify_mds_dataset.py` | — | Re-checks labels, mask presence, decoding and bytes against the originals |
| all | `run_pipeline.py` | `pipeline.yml` | Runs 1 → 2 → 3, with a y/n prompt if validate found failures |

Failures that validate reports:

| reason | meaning | usual fix |
|---|---|---|
| `unreadable`, `image-decode-invalid` | image can't be read or is truncated | remediate |
| `missing-mask` | no mask matched this tampered image | add the dataset's naming suffix to `mask_suffixes`; don't delete |
| `ambiguous-mask` | more than one mask matched (e.g. `x.png` and `x.bmp`) | remove the duplicate mask |
| `mask-decode-invalid` | mask can't be decoded | remediate |
| `mask-size-mismatch` | mask and image have different sizes | check the dataset release; remediate |
| `empty-mask` | tampered image whose mask marks nothing | remediate |

`remediate_dataset.yml` leaves `missing-mask` out of `reasons:` on purpose. It
is almost always a naming mismatch, not bad data. The converter never writes a
tampered image without its mask, even with `validate: false`.

## Running it (office PC)

```bash
pip install mosaicml-streaming pillow numpy tqdm pyyaml   # or: bash scripts/install_deps.sh

# 1. edit configs/mds/datasets.yml: data_root, folder names, splits
#    and the TODO paths in validate_dataset.yml / remediate_dataset.yml / mds_dataset.yml

# 2. validate only first, then read the report's "failed by dataset" summary
python packages/mdsconverter/run_pipeline.py --config configs/mds/pipeline.yml --only validate

#    lots of missing-mask in one dataset? -> look at its mask filenames and
#    add the suffix to mask_suffixes, then re-run validate

# 3. smoke test: 500 samples per split to a scratch output
python packages/mdsconverter/build_mds_dataset.py --config configs/mds/mds_dataset.yml \
    --limit 500 --out D:/forensics_mds_smoke

# 4. full run (remediate asks y/n if validate found failures)
python packages/mdsconverter/run_pipeline.py --config configs/mds/pipeline.yml

# 5. check each split
python packages/mdsconverter/verify_mds_dataset.py --mds D:/forensics_mds/train \
    --decode-check 500 --spot-check 500 --source-root D:/forensics_raw
python packages/mdsconverter/verify_mds_dataset.py --mds D:/forensics_mds/test \
    --decode-check 500 --spot-check 500 --source-root D:/forensics_raw
```

Every config key can be overridden on the command line (`--num-workers 12`,
`--split test`, `--only-datasets Columbia`, ...). Run a script with `--help` to
see them all.

## Training on it

In the training yml ([configs/normal/train_forensics.yml](../../configs/normal/train_forensics.yml)):

```yaml
data_format: mds
datasets:
  train:
    dataroot: D:/forensics_mds/train
  test:
    dataroot: D:/forensics_mds/test     # a list of MDS dirs also works
```

## Stored columns

| column | type | notes |
|---|---|---|
| `image` | bytes | original file bytes, **never re-encoded** (keeps JPEG compression traces intact) |
| `mask` | bytes | original mask file bytes; empty for authentic images |
| `label` / `label_str` | int / str | 0 `authentic`, 1 `tampered` |
| `dataset`, `split` | str | from `datasets.yml` |
| `orig_path`, `mask_orig_path` | str | relative to `data_root`, `/`-separated |
| `ext`, `mask_ext`, `width`, `height`, `filesize_bytes` | | |

## Windows notes

- `MDSWriter` reads `D:\...` as a URL with scheme `d:` and rejects it.
  `build_mds_dataset.py` works around this by changing into the output's
  parent folder during the write, so pass normal absolute paths.
- `ForensicsMDSDataset` reads shards with per-shard `MDSReader`s rather than
  `StreamingDataset`. `StreamingDataset`'s shared-memory setup fails inside
  spawned DataLoader workers on Windows.
- Output folders must not already contain an MDS dataset. Delete
  `<out>/<split>` before rebuilding it.
