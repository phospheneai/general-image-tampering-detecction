# general-image-tampering-detecction

A config-driven training framework for the Authenta general image forgery
segmentation model: a DINOv3 ViT-L/16 + LoRA backbone with a lightweight
convolutional head that predicts a per-pixel binary forgery mask
(`B x 1 x H x W`).

The training data is **processed-v1** — 1.86M images from 15 public forgery
datasets, stored in S3 as MosaicML Streaming (MDS) shards and loaded directly
by PyTorch, from S3 or from a local copy.

**New here? Follow [Quick start](#quick-start) top to bottom** — it takes you
from nothing to loading the dataset, getting the model weights and training.
Everything after it is reference.

## Contents

- [Quick start](#quick-start)
- [1. The dataset (processed-v1)](#1-the-dataset-processed-v1)
- [2. The model](#2-the-model)
- [3. Train, evaluate, and debug](#3-train-evaluate-and-debug)
- [4. About the repo](#4-about-the-repo)
- [5. SageMaker training](#5-sagemaker-training)
- [6. Troubleshooting](#6-troubleshooting)
- [7. Contributing](#7-contributing)

---

## Quick start

Run every command from the repo folder. Each step says what you should see.

### Step 0 — what you need

| | why |
|---|---|
| Linux (or macOS), `git`, the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) | |
| **Python 3.11** (3.12 also works; 3.13+ does not — `numpy==1.26.4` is pinned) | |
| GitHub access to `phospheneai/general-image-tampering-detecction` | private repo |
| AWS credentials with `s3:GetObject` (+ `s3:ListBucket`) on `s3://authenta-data-rnd/image-tampering-detection/processed-v1/` | the dataset |
| A Hugging Face account that has accepted the [DINOv3 licence](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | the backbone weights |
| An NVIDIA GPU | training only — loading data works on CPU |

### Step 1 — get the code

```bash
git clone https://github.com/phospheneai/general-image-tampering-detecction.git
cd general-image-tampering-detecction
git checkout processed-v1-dataset    # until it is merged into the default branch
```

GitHub asks for your username and a personal access token (not your password).

### Step 2 — install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision
bash scripts/install_deps.sh
```

No `python3.11`? Replace the first line with
`pip install uv && uv venv --python 3.11 .venv`.
On a machine with CUDA, install the matching torch build from
[pytorch.org](https://pytorch.org/get-started/locally/) instead of the plain
`pip install torch torchvision`.

`install_deps.sh` ends with `Authenta dependencies installed successfully.`
Your prompt now starts with `(.venv)` — run `source .venv/bin/activate` again in
every new terminal.

### Step 3 — AWS access to the dataset

Create a profile (it asks for your access key and secret — type them into the
terminal only; never paste them into chat or commit them):

```bash
aws configure --profile <your-profile>
export AWS_PROFILE=<your-profile>
```

Check it:

```bash
aws s3 ls s3://authenta-data-rnd/image-tampering-detection/processed-v1/
```

Expected: `PRE test/`, `PRE train/` and `train_failed.csv`. `AccessDenied` means
your credentials lack the permissions in Step 0.

> On the team EC2 instance the machine's own role (`instanceRole-s3-access`)
> **cannot** read this bucket yet — always `export AWS_PROFILE=...` there.

### Step 4 — get the model weights (DINOv3 backbone)

```bash
hf auth login                        # paste a Hugging Face read token when asked
bash scripts/download_artifacts.sh
```

Expected: `[done] DINOv3 ViT-L/16 artifact downloaded successfully.` and the
files `artifacts/mirror/dinov3-vitl16/config.json` + `model.safetensors`
(~1.2 GB, not committed to Git). Only needed for training/evaluation — loading
the dataset works without it.

### Step 5 — the dataset: stream it or download it

**Option A — stream from S3 (recommended, nothing to download up front).**
Shards are fetched on demand into a local cache folder. Nothing to do now; the
configs and code below take the S3 path.

**Option B — download a local copy** (1.14 TB; train alone is 1.13 TB, test 15 GB):

```bash
aws s3 sync s3://authenta-data-rnd/image-tampering-detection/processed-v1/ /data/processed-v1/ --only-show-errors
```

On the team EC2 instance a local copy already exists at
`/home/ubuntu/data/processed/processed-v1/`.

### Step 6 — check that the dataset loads in PyTorch

```bash
python tests/smoke_test_s3.py
```

~30 seconds, downloads 1 GB into a temp folder and deletes it. Expected (label
lists vary):

```
S3 access OK as arn:aws:iam::...:user/<you>

[train] streaming s3://authenta-data-rnd/image-tampering-detection/processed-v1/train  ...
  torch Dataset: yes | samples: 1,827,437 (expected 1,827,437)
  sample 0: LAION-Mobile / authentic / LAION-Mobile/images/authentic/180493.jpeg (3264x2448 original)
  authentic sample 0: label=0, mask all zero ✓
  tampered  sample 1: label=1, mask has tampered pixels ✓
  batch 0: image=(8, 3, 512, 512), mask=(8, 1, 512, 512), edge_mask=(8, 1, 512, 512), label=(8,)  labels=[0, 1, 0, 1, ...]
  ...
[test] ...
  torch Dataset: yes | samples: 27,604 (expected 27,604)
  ...
PASS
```

If you have a local copy (Option B / the EC2 instance), also run the check of
every loading path — local and S3, direct and via the training config, 0 and 4
workers, resume mid-epoch (~5 min):

```bash
python tests/smoke_test_all_paths.py --local-root /home/ubuntu/data/processed/processed-v1
```

Ends with `12/12 passed` and `PASS`.

### Step 7 — point the training config at the dataset

Open [`configs/normal/train_forensics_mds.yml`](configs/normal/train_forensics_mds.yml)
and set the two `dataroot` entries and `cache_dir`:

```yaml
data_format: mds
cache_dir: /data/mds_cache            # Option A: where streamed shards are cached (needs ~1.2 TB free for a full epoch)

datasets:
  train:
    dataroot:
      - s3://authenta-data-rnd/image-tampering-detection/processed-v1/train   # Option A
      # - /data/processed-v1/train                                            # Option B
  test:
    dataroot:
      - s3://authenta-data-rnd/image-tampering-detection/processed-v1/test
      # - /data/processed-v1/test
```

The file ships pointing at the EC2 local copy
(`/home/ubuntu/data/processed/processed-v1/...`). Check it loads:

```bash
python -c "
from authgenforge.options.load import get_dataloaders_from_yml
tl, vl = get_dataloaders_from_yml('configs/normal/train_forensics_mds.yml')
print(len(tl.dataset), len(vl.dataset), {k: tuple(v.shape) for k, v in next(iter(vl)).items()})"
```

Expected: `1827437 27604 {'image': (4, 3, 512, 512), 'mask': (4, 1, 512, 512), 'edge_mask': (4, 1, 512, 512), 'label': (4,)}`

### Step 8 — train

A few real training steps first (needs Step 4; GPU recommended, `--cpu` works but is slow):

```bash
python tests/smoke_test_pipeline.py --config configs/normal/train_forensics_mds.yml
```

Then the real run:

```bash
python notebooks/train.py --config configs/normal/train_forensics_mds.yml --end_epoch 10
```

Checkpoints and logs go to `checkpoints/train_forensics_mds_v1/`
([layout](#checkpointoutput-layout)).

### Step 9 — evaluate

```bash
python notebooks/eval.py --config configs/normal/train_forensics_mds.yml
```

Uses `eval_settings.checkpoint_path` (the best checkpoint from Step 8) on the
test split and reports pixel-level IoU, F1, precision, recall and accuracy.

---

## 1. The dataset (processed-v1)

| | |
|---|---|
| **S3** | `s3://authenta-data-rnd/image-tampering-detection/processed-v1/{train,test}/` |
| **Local copy (team EC2)** | `/home/ubuntu/data/processed/processed-v1/` |
| **Built from** | `s3://authenta-general-image-forgery-dataset/dataset-v1/` |
| **Format** | MDS: per split an `index.json` + 512 MB `shard.NNNNN.mds` files |
| **train** | 1,827,437 images — 1,003,616 authentic / 823,821 tampered (55:45) — 8 datasets, 2,108 shards, 1,128.9 GB |
| **test** | 27,604 images — 6,329 authentic / 21,275 tampered — 7 datasets, 28 shards, 14.6 GB |
| **Excluded** | 1,204 broken raw files (truncated images, empty/mis-sized masks), listed in `processed-v1/train_failed.csv` |

Per-dataset counts, sizes and splits: [DATASETS.md](DATASETS.md).

**What is stored.** Each sample holds the **original image and mask file bytes**
(never re-encoded — JPEG compression traces are part of the signal) plus
`label` (0 authentic / 1 tampered), `label_str`, `dataset`, `split`,
`orig_path`, `mask_orig_path`, `width`, `height`, `ext`, `filesize_bytes`.
Authentic images have no mask stored (an all-zero mask is built when loading).
Shards are shuffled and balanced: every shard has the split's authentic:tampered
ratio, with datasets mixed.

**What PyTorch gets.** [`ForensicsMDSDataset`](authgenforge/data/forensics_mds_dataset.py)
is a regular `torch.utils.data.Dataset`; each sample is

| key | type / shape | meaning |
|---|---|---|
| `image` | float32 `(3, 512, 512)` | RGB, 0–1 |
| `mask` | float32 `(1, 512, 512)` | 1 = tampered pixel, all zero for authentic |
| `edge_mask` | float32 `(1, 512, 512)` | band around the tampered region's boundary (used by the loss) |
| `label` | int64 | 0 = authentic, 1 = tampered |

512 is the training crop (`crop_size` in the config), not the stored size —
change it freely, no rebuild needed.

**Load it yourself:**

```python
from torch.utils.data import DataLoader
from authgenforge.data.forensics_mds_dataset import ForensicsMDSDataset
from authgenforge.augmentations.presets import get_train_transforms, get_val_transforms

# from S3 (shards cached under cache_dir as they are read) ...
ds = ForensicsMDSDataset("s3://authenta-data-rnd/image-tampering-detection/processed-v1/train",
                         transform=get_train_transforms(crop_size=512),
                         cache_dir="/data/mds_cache")
# ... or from a local copy
# ds = ForensicsMDSDataset("/data/processed-v1/train", transform=get_train_transforms(crop_size=512))

print(len(ds))              # 1827437
print(ds.get_raw(0))        # stored bytes + metadata: label_str, dataset, orig_path, width, height, ...
loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
batch = next(iter(loader))  # image (8,3,512,512), mask (8,1,512,512), edge_mask (8,1,512,512), label (8,)
```

Use `get_val_transforms` (deterministic center crop) for the test split. A list
of roots (e.g. train + test, or several S3 paths) is also accepted.

> **Streaming + `shuffle=True`:** each random sample can pull a different
> 512 MB shard, so the first batches are slow and a full epoch caches the whole
> split (~1.13 TB for train) under `cache_dir`. Later epochs read from the cache.
> For a quick look use `shuffle=False`.

**How it was built / rebuilding.** `packages/mdsconverter/` turns the raw
dataset into processed-v1 (prepare → validate → remediate → convert → verify).
Details: [MDS_DATASET.md](MDS_DATASET.md) (format, columns, verification) and
[DATASET_PIPELINE.md](DATASET_PIPELINE.md) (the stages and their configs in
`configs/mds/`). Short version, on a machine with the raw data:

```bash
python packages/mdsconverter/prepare_compraise.py      # one-off: unzip compRAISE
python packages/mdsconverter/prepare_columbia.py       # one-off: Columbia edgemasks -> binary masks
python packages/mdsconverter/run_pipeline.py --config configs/mds/pipeline.yml
python packages/mdsconverter/verify_mds_dataset.py --mds <out>/train --decode-check 500 --spot-check 500 \
    --source-root /home/ubuntu/data/raw/train /home/ubuntu/data/extracted
aws s3 sync <out>/ s3://authenta-data-rnd/image-tampering-detection/processed-v1/ --only-show-errors
```

## 2. The model

| | |
|---|---|
| **Backbone** | DINOv3 ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`), frozen, LoRA on `q_proj`/`k_proj`/`v_proj` |
| **Head** | 3 convolutions (1024 → 512 → 256 → 1), bilinear-upsampled to the input size |
| **Output** | per-pixel forgery logits `B x 1 x H x W` |
| **Loss** | `pixel_bce + edge_lambda * edge_bce` (edge-weighted toward forgery boundaries, `edge_lambda=20`) |
| **Code** | [`authgenforge/networks/dinov3_segmentation.py`](authgenforge/networks/dinov3_segmentation.py) |

**Weights.** The backbone comes from Hugging Face via
`bash scripts/download_artifacts.sh` (Quick start Step 4) into
`artifacts/mirror/dinov3-vitl16/`; configs point at it with
`structure.backbone.model_path`. Weights are never committed to Git.

**Trained checkpoints** are written by training to `checkpoints/<name>/`
(`<name>_best.pth` = best validation IoU). To start from or evaluate an existing
checkpoint, set `pretraining_settings.checkpoint_path` (fine-tune from it) or
`eval_settings.checkpoint_path` (evaluate it) in the config.

## 3. Train, evaluate, and debug

Configs: [`configs/normal/train_forensics_mds.yml`](configs/normal/train_forensics_mds.yml)
(processed-v1, `data_format: mds`) and
[`configs/normal/train_forensics.yml`](configs/normal/train_forensics.yml)
(loose `images/` + `masks/` folders, `data_format: folder`). Fields you usually
touch:

- `datasets.train.dataroot` / `datasets.test.dataroot` (+ `cache_dir` for S3)
- `structure.backbone.model_path` — the DINOv3 directory
- `datasets.train.batch_size` — GPU memory; offset with `train_settings.grad_accum_steps`
- `datasets.train.crop_size` — training crop (default 512)
- `epoch_settings.total_epochs`

### Train

```bash
python notebooks/train.py --config configs/normal/train_forensics_mds.yml --end_epoch 10
```

or inline:

```python
from authgenforge.options.load import load_pipeline_from_yml

train_loader, test_loader, model, trainer = load_pipeline_from_yml(
    "configs/normal/train_forensics_mds.yml"
)
trainer.train_model(end_epoch=10)
```

`trainer.ckpt_dir` and `trainer.log_dir` print the exact output paths at startup.

### Resume a run

Set `train_settings.load_checkpoint_file_path` to a checkpoint file and
re-launch the same config (the path resolves relative to the yml).
`train_settings.resume_dataloader: true` (with `datasets.train.stateful_loader: true`)
continues at the exact shuffle position instead of restarting the epoch.

### Evaluate

```bash
python notebooks/eval.py --config configs/normal/train_forensics_mds.yml
```

or `from authgenforge.evals.evaluator import evaluate_from_yml; evaluate_from_yml("configs/normal/train_forensics_mds.yml")`.
Reads `eval_settings.checkpoint_path`, runs over `datasets.test` at a fixed
threshold, reports pixel-level TP/TN/FP/FN, IoU, F1, precision, recall, accuracy.

### Tests and debugging

| command | what it checks | needs |
|---|---|---|
| `python tests/smoke_test_s3.py` | processed-v1 loads from S3 as a torch Dataset (counts, labels, masks, batches) | AWS access |
| `python tests/smoke_test_all_paths.py` | all 12 ways the data can be loaded (local + S3, config, workers, resume, both backends); `--skip-s3` to run offline | local copy |
| `python tests/smoke_test_mds.py` | the converter, on a tiny generated dataset | nothing |
| `python tests/smoke_test_pipeline.py --config <yml>` | a few real training steps: data, forward/backward, checkpoint, inference (`--cpu` without GPU) | model weights |
| `python tests/overfit_single_batch.py --config <yml>` | the model can drive loss to ~0 on one batch — run before debugging a run that won't converge | model weights |

## 4. About the repo

### Repo layout

```
authgenforge/
  networks/      DINOv3 ViT-L/16 + LoRA + segmentation head
  losses/        pixel BCE + edge-weighted BCE (forgery segmentation loss)
  data/          dataset + dataloader builders (folder and MDS backends)
  augmentations/ PIL/OpenCV-based paired image+mask augmentation pipelines
  optimizers/    optimizer (layer-decay AdamW) + LR scheduler
  options/       yml -> pipeline builders
  training/      SegmentationTrainer (train/validate/checkpoint/resume)
  evals/         post-hoc evaluation over a held-out dataset
  utils/         logger, meters, small tensor/plot utilities
packages/
  mdsconverter/  raw dataset -> processed-v1 MDS shards (see section 1)
configs/
  normal/        training configs
  mds/           dataset-conversion configs
sagemaker/       SageMaker-specific training entry point (section 5)
notebooks/       thin launchers (train.py, eval.py)
scripts/         install / artifact download / setup
tests/           smoke tests + single-batch overfit check
checkpoints/     training output (not in Git)
```

| where | what |
|---|---|
| `authgenforge/data/forensics_mds_dataset.py` | `ForensicsMDSDataset` — processed-v1 as a torch `Dataset` (local or `s3://`) |
| `authgenforge/data/forensics_mds_dataloader.py` | `build_mds_dataloaders` — train/test loaders for `data_format: mds` |
| `authgenforge/data/forensics_dataset.py`, `dataloader.py` | the folder backend (`data_format: folder`) |
| `authgenforge/options/load.py` | pipeline builder; `_DATASET_BACKENDS` maps `data_format` to a loader |
| `authgenforge/options/option_utils.py` | yml parsing + path resolution |
| `authgenforge/networks/dinov3_segmentation.py` | `DINOv3Segmentation`, `build_dinov3_segmentation`, `load_checkpoint` |
| `authgenforge/losses/` | `PixelBCEWithLogitsLoss`, `EdgeWeightedBCEWithLogitsLoss`, `ForgerySegmentationLoss` |

This project has a single fixed architecture (DINOv3 ViT-L/16 + LoRA, binary
segmentation), so there is one loader (`load.py`); swapping the backbone or loss
is a deliberate architecture change, not a config flip. The dataset backend is
the one config switch (`data_format:`).

### Config schema

```yaml
name: train_forensics_mds_v1
model: dinov3_forensics_lora
type: binary_segmentation
data_context: normal           # normal | pdf — which configs/ subfolder this belongs to
data_format: mds               # mds | folder
cache_dir: ...                 # mds only: local cache for s3:// dataroots
cache_limit: null

datasets:
  train: {dataroot, n_workers, batch_size, crop_size, buffer_size, stateful_loader, pin_memory}
  test:  {dataroot, n_workers, crop_size, pin_memory}

structure:
  backbone:
    model_path: ""             # local HF dir: config.json + model.safetensors
    model_type: dinov3_vitl16
    lora_rank: 32
    lora_alpha: 64
    lora_dropout: 0.0

pretraining_settings:           # optional: start from a previously trained checkpoint
  want_load, checkpoint_path, strict_load

epoch_settings:
  total_epochs: N

train_settings:
  base_lr, weight_decay, layer_decay
  scheduler_type, warmup_epochs, warmup_lr, min_lr, T_0_epochs
  criterion: forgery_segmentation
  forgery_segmentation_loss: {edge_lambda}
  grad_accum_steps, grad_clip, save_interval, log_interval
  mixed_precision
  save_checkpoint_folder_path, load_checkpoint_file_path, resume_dataloader

eval_settings:                   # consumed by authgenforge/evals/evaluator.py
  checkpoint_path, image_size, threshold, batch_size, num_workers
  half_precision, output_dir
```

How it is parsed (`authgenforge/options/option_utils.py`'s `parse_yml`):

- **Paths resolve relative to the yml file**, not the working directory —
  `dataroot`, `structure.backbone.model_path`, `pretraining_settings.checkpoint_path`,
  `train_settings.{save_checkpoint_folder_path,load_checkpoint_file_path}`,
  `eval_settings.{checkpoint_path,output_dir}`. URLs (`s3://...`) are left as is.
- **Missing keys return `None`, not `KeyError`** (`NoneDict`); `load.py`'s
  `_cfg(value, default)` falls back only when a value is unset, so an explicit
  `0` or `False` is respected.

### Checkpoint/output layout

```
checkpoints/{experiment_name}/
  latest_checkpoint.pth       full state: model, optimizer, scheduler, scaler,
                               global_step, best_iou
  {experiment_name}_best.pth  copy of latest_checkpoint.pth at the best val IoU so far
  epoch{N}.pth                 end-of-epoch, model-only snapshot
  {run_tag}/                   per-run-day logs/plots/predictions
    logs/log.txt
    plots/
    predictions/
  eval/                        authgenforge/evals/evaluator.py output (default)
```

### Version pins

`install_deps.sh` pins a few versions for non-obvious reasons:

| Pin | Why |
|---|---|
| `numpy==1.26.4` | Lambda Stack's torch/torchvision are built against the NumPy 1.x ABI. NumPy 2.x breaks `torch.from_numpy()`/`.numpy()`. Also why Python must be ≤ 3.12. |
| `opencv-python-headless==4.11.0.86` | Kept below the release that forces `numpy>=2`. |
| `transformers==4.57.1` / `peft==0.19.1` | The exact pair the DINOv3 + LoRA loading path is verified against; a newer `peft` can need `transformers` internals an older one lacks — install them together. |
| `mosaicml-streaming==0.13.0` | Reads/writes the MDS shards; needs `numpy<2.2`. |

## 5. SageMaker training

For running as a managed SageMaker Training job (S3 data, spot-interruptible,
checkpoints synced to S3) instead of a bare GPU box, see
[`sagemaker/README.md`](sagemaker/README.md). The canonical training code stays
under `authgenforge/`; `sagemaker/` is an environment-specific addition.

## 6. Troubleshooting

| you see | cause | fix |
|---|---|---|
| `403 Forbidden`, `AccessDenied`, or `index.json not found!` | no S3 credentials, or the EC2 instance role is used | `export AWS_PROFILE=<your-profile>` (Quick start Step 3) |
| `ModuleNotFoundError: No module named 'authgenforge'` | not in the repo folder, or the environment isn't active | `cd` into the repo; `source .venv/bin/activate` |
| `ModuleNotFoundError: 'streaming'` / `'torchdata'` / `'peft'` | dependencies not installed | `bash scripts/install_deps.sh` |
| numpy fails to build during install | Python 3.13 or newer | use Python 3.11 (Step 2) |
| `Hugging Face CLI 'hf' was not found` / `401` / `gated repo` | not logged in, or DINOv3 licence not accepted | `hf auth login`; accept the licence on the model page (Step 0) |
| `UserWarning: 'set_vital' is deprecated` / `pin_memory ... no accelerator` | library notice / no GPU | harmless — ignore |
| first batches from S3 take minutes | `shuffle=True` touching many shards | expected on the first epoch; `shuffle=False` for quick checks |
| `No space left on device` while streaming | the cache grows to the full split size | point `cache_dir` at a disk with ~1.2 TB free, or set `cache_limit` |

## 7. Contributing

Before opening a PR:

- Run `tests/smoke_test_pipeline.py`. It catches import/wiring breaks across the
  whole pipeline in a couple of minutes.
- If you touched the model or a loss, run `tests/overfit_single_batch.py` — it
  should still drive loss to (near-)zero on one frozen batch.
- If you touched `authgenforge/data/` or `packages/mdsconverter/`, run
  `tests/smoke_test_mds.py` and `tests/smoke_test_all_paths.py`.
- A new config key that holds a file path is resolved in `option_utils.py`'s
  `parse_yml`, not ad hoc at the call site.

Conventions:

- **Keep diffs minimal.** Small, focused modules over premature abstraction.
- **Comments explain why, not what.** Hidden constraints, workarounds, pin reasons.
- **One data format per backend.** A new dataset format is a new sibling module
  in `authgenforge/data/` plus one entry in `_DATASET_BACKENDS`.
