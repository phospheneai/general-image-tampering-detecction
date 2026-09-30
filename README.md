# General Image Tampering Detection

This project trains the Authenta general image forgery segmentation model — a DINOv3 ViT-L/16 + LoRA backbone with a convolutional head that predicts a per-pixel forgery mask — on **processed-v1**, a 1.86M-image forgery dataset stored in S3 as MosaicML Streaming (MDS) shards.

It loads the dataset straight into PyTorch (from S3 or a local copy), downloads the model weights, trains, evaluates, and contains the pipeline that built the dataset from the raw sources.

## Features

- **processed-v1 dataset in S3** — 1,827,437 train / 27,604 test images from 15 public forgery datasets, original image and mask bytes (never re-encoded), shuffled and authentic:tampered-balanced 512 MB shards
- **PyTorch `Dataset`** (`ForensicsMDSDataset`) that streams shards from S3 on demand or reads a local copy, and works with multi-worker and resumable (`StatefulDataLoader`) loaders
- Each sample gives `image`, `mask`, `edge_mask` and `label` (0 authentic / 1 tampered)
- Config-driven training and evaluation of the DINOv3 ViT-L/16 + LoRA segmentation model
- Dataset conversion pipeline (`packages/mdsconverter/`): prepare → validate → convert → verify, raw folders → MDS shards
- Smoke tests that prove the dataset loads (from S3 and locally) and the pipeline trains

## Project Structure

```text
.
├── authgenforge/
│   ├── data/
│   │   ├── forensics_mds_dataset.py     # ForensicsMDSDataset — processed-v1 as a torch Dataset
│   │   ├── forensics_mds_dataloader.py  # train/test DataLoaders for data_format: mds
│   │   └── forensics_dataset.py, dataloader.py   # loose images/ + masks/ folders (data_format: folder)
│   ├── networks/        # DINOv3 ViT-L/16 + LoRA + segmentation head
│   ├── losses/          # pixel BCE + edge-weighted BCE
│   ├── augmentations/   # paired image + mask transforms (512 crop)
│   ├── options/         # yml -> dataloaders, model, trainer (load.py)
│   ├── training/        # SegmentationTrainer
│   └── evals/           # evaluation
├── packages/mdsconverter/   # raw dataset -> processed-v1 MDS shards
├── configs/
│   ├── normal/train_forensics_mds.yml   # training config for processed-v1
│   ├── normal/train_forensics.yml       # training config for loose folders
│   └── mds/                             # dataset-conversion configs
├── notebooks/           # train.py, eval.py launchers
├── scripts/             # install_deps.sh, download_artifacts.sh
├── tests/               # smoke_test_s3.py, smoke_test_all_paths.py, smoke_test_mds.py, smoke_test_pipeline.py
├── sagemaker/           # SageMaker training entry point (see sagemaker/README.md)
├── DATASETS.md          # per-dataset counts, sizes and splits
├── MDS_DATASET.md       # dataset format and verification details
├── DATASET_PIPELINE.md  # how the conversion pipeline is run
└── README.md
```

## Requirements

- **Python 3.11** (3.12 also works; 3.13+ does not — `numpy==1.26.4` is pinned)
- `git` and the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
- GitHub access to this private repo
- AWS credentials with `s3:GetObject` and `s3:ListBucket` on `s3://authenta-data-rnd/image-tampering-detection/processed-v1/`
- A Hugging Face account that has accepted the [DINOv3 licence](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) (model weights)
- An NVIDIA GPU for training (loading the dataset works on CPU)

Get the code and install the dependencies:

```bash
git clone https://github.com/phospheneai/general-image-tampering-detecction.git
cd general-image-tampering-detecction
git checkout processed-v1-dataset        # until it is merged into the default branch

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision            # or the CUDA build from pytorch.org
bash scripts/install_deps.sh             # ends with "Authenta dependencies installed successfully."
```

No `python3.11`? Use `pip install uv && uv venv --python 3.11 .venv` instead of the `python3.11 -m venv` line. Run `source .venv/bin/activate` in every new terminal.

## Configuration

The training configuration for processed-v1 is [configs/normal/train_forensics_mds.yml](configs/normal/train_forensics_mds.yml).

Key settings include:

- `data_format`: `mds` for processed-v1 shards (`folder` for loose `images/` + `masks/` folders)
- `datasets.train.dataroot` / `datasets.test.dataroot`: an S3 URL or a local folder per split
- `cache_dir`: where shards streamed from S3 are cached (a full train epoch needs ~1.2 TB free)
- `datasets.train.batch_size`, `n_workers`, `crop_size`: loader settings (images are cropped to `crop_size` when loaded; stored images are full size)
- `structure.backbone.model_path`: the DINOv3 weights folder
- `epoch_settings.total_epochs`, `train_settings.*`: training schedule, learning rate, loss
- `eval_settings.checkpoint_path`: the checkpoint to evaluate

Example:

```yaml
data_format: mds
cache_dir: /data/mds_cache

datasets:
  train:
    dataroot:
      - s3://authenta-data-rnd/image-tampering-detection/processed-v1/train
      # - /home/ubuntu/data/processed/processed-v1/train     # local copy instead
    n_workers: 4
    batch_size: 4
    crop_size: 512
  test:
    dataroot:
      - s3://authenta-data-rnd/image-tampering-detection/processed-v1/test
    n_workers: 2
    crop_size: 512

structure:
  backbone:
    model_path: ../../artifacts/mirror/dinov3-vitl16
```

The file ships pointing at the local copy on the team EC2 instance (`/home/ubuntu/data/processed/processed-v1/`). Paths resolve relative to the yml file; `s3://` URLs are used as-is. The dataset-conversion settings are in [configs/mds/](configs/mds/) (see [DATASET_PIPELINE.md](DATASET_PIPELINE.md)).

## Usage

From the repository root, with the environment active:

**1. Set your AWS profile** (create it once with `aws configure --profile <your-profile>`; type the keys into the terminal only)

```bash
export AWS_PROFILE=<your-profile>
aws s3 ls s3://authenta-data-rnd/image-tampering-detection/processed-v1/
# expected:  PRE test/   PRE train/   train_failed.csv
```

**2. Check the dataset loads in PyTorch from S3** (~30 s, downloads 1 GB to a temp folder and deletes it)

```bash
python tests/smoke_test_s3.py
# ... torch Dataset: yes | samples: 1,827,437 ... batch 0: image=(8, 3, 512, 512), mask=(8, 1, 512, 512), edge_mask=(8, 1, 512, 512), label=(8,) ... PASS
```

With a local copy, also check every loading path (local + S3, training config, workers, resume):

```bash
python tests/smoke_test_all_paths.py --local-root /home/ubuntu/data/processed/processed-v1
# ... 12/12 passed  PASS
```

**3. Load it in your own code**

```python
from torch.utils.data import DataLoader
from authgenforge.data.forensics_mds_dataset import ForensicsMDSDataset
from authgenforge.augmentations.presets import get_train_transforms

ds = ForensicsMDSDataset("s3://authenta-data-rnd/image-tampering-detection/processed-v1/train",
                         transform=get_train_transforms(crop_size=512),
                         cache_dir="/data/mds_cache")   # or a local folder, without cache_dir
loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
batch = next(iter(loader))    # image (8,3,512,512), mask (8,1,512,512), edge_mask (8,1,512,512), label (8,)
print(ds.get_raw(0)["label_str"], ds.get_raw(0)["dataset"])   # stored metadata of a sample
```

**4. Download the model weights** (DINOv3 ViT-L/16 backbone, ~1.2 GB)

```bash
hf auth login                          # paste a Hugging Face read token
bash scripts/download_artifacts.sh     # -> artifacts/mirror/dinov3-vitl16/{config.json, model.safetensors}
```

**5. Train**

```bash
python tests/smoke_test_pipeline.py --config configs/normal/train_forensics_mds.yml   # a few real steps first
python notebooks/train.py --config configs/normal/train_forensics_mds.yml --end_epoch 10
```

**6. Evaluate**

```bash
python notebooks/eval.py --config configs/normal/train_forensics_mds.yml
```

This will:

1. Connect to S3 with your profile and confirm access to processed-v1
2. Stream the dataset (only the shards that are read) and check it loads as a PyTorch `Dataset`
3. Download the DINOv3 backbone weights from Hugging Face
4. Build the DataLoaders, model, loss and optimizer from the yml and train, saving checkpoints
5. Evaluate the best checkpoint on the test split (pixel-level IoU, F1, precision, recall, accuracy)

To rebuild processed-v1 from the raw dataset instead, see [DATASET_PIPELINE.md](DATASET_PIPELINE.md) (`python packages/mdsconverter/run_pipeline.py --config configs/mds/pipeline.yml`).

## Output

The dataset in S3 (a local copy has the same layout):

```text
s3://authenta-data-rnd/image-tampering-detection/processed-v1/
├── train/
│   ├── index.json
│   └── shard.00000.mds … shard.02107.mds    # 2,108 × 512 MB, 1,827,437 samples
├── test/
│   ├── index.json
│   └── shard.00000.mds … shard.00027.mds    # 28 × 512 MB, 27,604 samples
└── train_failed.csv                         # 1,204 broken raw files left out, with reasons
```

Each sample returned by `ForensicsMDSDataset`:

| key | type / shape | meaning |
|---|---|---|
| `image` | float32 `(3, 512, 512)` | RGB, values 0–1 |
| `mask` | float32 `(1, 512, 512)` | 1 = tampered pixel; all zero for authentic |
| `edge_mask` | float32 `(1, 512, 512)` | band around the tampered region's boundary |
| `label` | int64 | 0 = authentic, 1 = tampered |

Training writes under `checkpoints/<name>/`:

```text
checkpoints/train_forensics_mds_v1/
├── latest_checkpoint.pth                 # full state, for resuming
├── train_forensics_mds_v1_best.pth       # best validation IoU — used by eval
├── epoch{N}.pth
└── <run_tag>/logs/, plots/, predictions/
```

## Notes

- **On the team EC2 instance the machine's own AWS role cannot read the bucket** — always `export AWS_PROFILE=...` first. Without it you get `403 Forbidden` or `index.json not found!`.
- Streaming with `shuffle=True` is slow at first (each random sample can pull a different 512 MB shard) and a full epoch caches the whole split under `cache_dir`; use `shuffle=False` for quick checks.
- `crop_size` can be changed freely (e.g. 384, 768) — stored images are full size, so no rebuild is needed.
- Per-dataset counts and sizes: [DATASETS.md](DATASETS.md). Format, columns and verification: [MDS_DATASET.md](MDS_DATASET.md). Columbia masks mark bright red (camera 1 near the splice boundary) as tampered.
- The model weights are not committed to Git; `download_artifacts.sh` fetches them. Trained checkpoints stay under `checkpoints/` (also not in Git).
- `UserWarning: 'set_vital' is deprecated` and `pin_memory ... no accelerator` are harmless.
- `ModuleNotFoundError`: run from the repo root with the environment active, and re-run `bash scripts/install_deps.sh` if a package is missing.
- For SageMaker training jobs see [sagemaker/README.md](sagemaker/README.md).
