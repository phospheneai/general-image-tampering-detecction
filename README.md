# General Image Tampering Detection

This project converts general image forgery datasets (images plus tampering masks) into a MosaicML StreamingDataset-compatible format and trains a forgery segmentation model on it.

It scans the dataset folders, pairs every tampered image with its mask, checks that each file decodes, writes shuffled streaming shards per split, verifies the output, and loads it as a PyTorch dataset for training.

## Features

- Scans one folder per source dataset (`images/authentic`, `images/tampered`, `masks/tampered`) and pairs each tampered image with its mask
- Assigns each dataset to the train or test split from a single config file
- Fully decodes every image and mask, skipping broken files and listing them in a CSV
- Stores the original image and mask bytes (never re-encoded) with the label and metadata
- Writes shuffled MosaicML StreamingDataset (MDS) shards with the same authentic:tampered ratio in every shard
- Verifies the generated dataset against the original files
- Loads the shards as a PyTorch `Dataset` (`image`, `mask`, `edge_mask`, `label`) from a local folder or from S3
- Trains and evaluates the DINOv3 ViT-L/16 + LoRA segmentation model from a yml config

## Project Structure

```text
.
├── authgenforge/
│   ├── data/
│   │   ├── forensics_mds_dataset.py     # PyTorch Dataset over the MDS shards
│   │   └── forensics_mds_dataloader.py  # train/test DataLoaders
│   ├── networks/                        # DINOv3 ViT-L/16 + LoRA segmentation model
│   ├── losses/
│   ├── augmentations/
│   ├── options/                         # builds the training pipeline from a yml
│   ├── training/
│   └── evals/
├── packages/mdsconverter/
│   ├── prepare_compraise.py             # one-off: unzip compRAISE
│   ├── prepare_columbia.py              # one-off: Columbia edgemasks -> binary masks
│   ├── validate_dataset.py
│   ├── remediate_dataset.py
│   ├── build_mds_dataset.py
│   ├── verify_mds_dataset.py
│   └── run_pipeline.py
├── configs/
│   ├── mds/
│   │   ├── datasets.yml                 # which datasets, where, which split
│   │   ├── mds_dataset.yml              # conversion settings
│   │   └── pipeline.yml
│   └── normal/
│       └── train_forensics_mds.yml      # training config
├── notebooks/                           # train.py, eval.py
├── scripts/                             # install_deps.sh, download_artifacts.sh
├── tests/
└── README.md
```

## Requirements

Python 3.11 is recommended (3.12 also works; `numpy==1.26.4` is pinned, so not 3.13+).

Install the dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch torchvision
bash scripts/install_deps.sh
```

The project uses the MosaicML Streaming library (`mosaicml-streaming`), installed by `install_deps.sh`.

For training, download the DINOv3 ViT-L/16 backbone (needs a Hugging Face account with access to the model):

```bash
hf auth login
bash scripts/download_artifacts.sh
```

## Configuration

The main configuration files are [configs/mds/datasets.yml](configs/mds/datasets.yml) and [configs/mds/mds_dataset.yml](configs/mds/mds_dataset.yml).

Key settings include:

- `data_root`: folder holding one subfolder per dataset
- `datasets`: each dataset and its split (`train` / `test` / `skip`), or `{split, path}` for a dataset stored elsewhere
- `mask_suffixes`: filename suffixes tried when pairing a tampered image with its mask
- `out`: output root; one MDS dataset per split is written to `<out>/<split>/`
- `shard_size_mb`: maximum size of each shard
- `num_workers`: number of worker processes
- `seed`: shuffle seed, for a reproducible build
- `validate`: fully decode every image and mask before writing it
- `limit`: cap the number of samples per split (for a quick test)

Example:

```yaml
# datasets.yml
data_root: "/path/to/raw/train"
datasets:
  CASIA_v2: train
  tampCOCO: train
  COCO2017_test: {split: test, path: "/path/to/raw/test/COCO2017"}
  IMD2020: {split: test, path: "/path/to/raw/test/IMD2020"}
mask_suffixes: ["", "_mask"]

# mds_dataset.yml
datasets_config: datasets.yml
out: "/path/to/processed/processed-v1"
shard_size_mb: 512
num_workers: 2
seed: 42
validate: true
limit: null
```

The training configuration is [configs/normal/train_forensics_mds.yml](configs/normal/train_forensics_mds.yml): set `datasets.train.dataroot` and `datasets.test.dataroot` to `<out>/train` and `<out>/test` (or `s3://...` paths plus `cache_dir`).

## Usage

From the repository root, run:

```bash
python packages/mdsconverter/prepare_compraise.py     # one-off, only for compRAISE
python packages/mdsconverter/prepare_columbia.py      # one-off, only for Columbia
python packages/mdsconverter/run_pipeline.py --config configs/mds/pipeline.yml
```

This will:

1. Load the YAML configuration
2. Scan the dataset folders and pair each tampered image with its mask
3. Decode every image and mask, skipping broken files
4. Shuffle each split and balance authentic and tampered samples across shards
5. Write MDS shards to `<out>/train/` and `<out>/test/`

Verify each split:

```bash
python packages/mdsconverter/verify_mds_dataset.py --mds /path/to/processed/processed-v1/train \
    --decode-check 500 --spot-check 500 --source-root /path/to/raw/train /path/to/extracted
```

(`--source-root` takes every folder the split's datasets came from, e.g. the raw folder plus the folder the prepare scripts wrote to.)

Load the dataset in PyTorch:

```python
from torch.utils.data import DataLoader
from authgenforge.data.forensics_mds_dataset import ForensicsMDSDataset
from authgenforge.augmentations.presets import get_train_transforms

ds = ForensicsMDSDataset("/path/to/processed/processed-v1/train",
                         transform=get_train_transforms(crop_size=512))
loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
batch = next(iter(loader))
```

To read the same shards from S3, pass the `s3://` path of the split and a local `cache_dir`.

Train and evaluate:

```bash
python notebooks/train.py --config configs/normal/train_forensics_mds.yml --end_epoch 10
python notebooks/eval.py --config configs/normal/train_forensics_mds.yml
```

Check that everything works:

```bash
python tests/smoke_test_mds.py                                               # the converter
python tests/smoke_test_all_paths.py --skip-s3 --local-root /path/to/processed/processed-v1   # loading
```

## Output

The converter writes one MDS dataset per split under the configured `out` directory.

Typical output structure:

```text
processed-v1/
├── train/
│   ├── index.json
│   └── shard.00000.mds …
├── test/
│   ├── index.json
│   └── shard.00000.mds …
└── train_failed.csv
```

If some images fail during processing, they are listed with the reason in `<split>_failed.csv`.

Each sample loaded by `ForensicsMDSDataset` contains:

| key | shape | meaning |
|---|---|---|
| `image` | `(3, 512, 512)` float32 | RGB image, values 0–1 |
| `mask` | `(1, 512, 512)` float32 | 1 = tampered pixel, all zero for authentic |
| `edge_mask` | `(1, 512, 512)` float32 | boundary band of the tampered region |
| `label` | int64 | 0 = authentic, 1 = tampered |

Training writes checkpoints and logs to `checkpoints/<name>/`.

## Notes

- The converter expects the source datasets to be accessible locally.
- Labels come from the folder names `authentic` and `tampered`.
- You may need to adjust the config paths to match your environment.
- Images are stored at full size; `crop_size` in the training config only sets the crop used when loading.
- processed-v1 (1,827,437 train / 27,604 test images) is available at `s3://authenta-data-rnd/image-tampering-detection/processed-v1/`; reading it needs AWS credentials with access to that bucket.
- Per-dataset counts: [DATASETS.md](DATASETS.md). Format and pipeline details: [MDS_DATASET.md](MDS_DATASET.md), [DATASET_PIPELINE.md](DATASET_PIPELINE.md).
