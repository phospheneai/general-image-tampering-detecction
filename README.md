# general-image-tampering-detecction

A config-driven training framework for the Authenta general image forgery
segmentation model.

It ships with a DINOv3 ViT-L/16 + LoRA backbone and a lightweight
convolutional segmentation head, predicting a dense per-pixel binary
forgery mask (`B x 1 x H x W`) rather than an image-level real/fake label.
The dataset backend is selected from YAML — loose image/mask folders, or the
processed MosaicML Streaming (MDS) shards built by `packages/mdsconverter/`
(see [MDS_DATASET.md](MDS_DATASET.md) and [DATASETS.md](DATASETS.md)).

## Contents

- [1. Installation](#1-installation)
- [2. About the repo](#2-about-the-repo)
- [3. Train, evaluate, and play around](#3-train-evaluate-and-play-around)
- [4. Contributing](#4-contributing)

## 1. Installation

```bash
bash scripts/install_deps.sh       # installs deps with every version pin that matters
bash scripts/download_artifacts.sh  # fetches the DINOv3 ViT-L/16 backbone
```

Or run `bash scripts/setup.sh`, which installs the dependencies and prepares
the artifact directory in one shot for a fresh session.

`install_deps.sh` pins a few versions for non-obvious reasons:

| Pin | Why |
|---|---|
| `numpy==1.26.4` | The system torch/torchvision builds are compiled against the NumPy 1.x ABI. NumPy 2.x breaks `torch.from_numpy()`/`.numpy()`. It also means Python ≤ 3.12. |
| `opencv-python-headless==4.11.0.86` | Kept below the release that forces `numpy>=2`. |
| `transformers==4.57.1` / `peft==0.19.1` | The exact pair the DINOv3 + LoRA loading path (`AutoModel.from_pretrained(..., local_files_only=True)` + `get_peft_model`) is verified against. A newer `peft` can require `transformers` internals (e.g. `HybridCache`) that an older pinned `transformers` doesn't export yet — install the two together, not independently. |
| `mosaicml-streaming==0.13.0` | Reads and writes the MDS shards. Needs `numpy<2.2`, compatible with the pin above. |

`download_artifacts.sh` fetches the official Hugging Face DINOv3 ViT-L/16
model (`facebook/dinov3-vitl16-pretrain-lvd1689m`) into
`artifacts/mirror/dinov3-vitl16/` — `config.json` + `model.safetensors`.
The model is gated: accept its licence on Hugging Face and run
`hf auth login` first. The weights are intentionally **not** committed to
Git; see [`.gitignore`](.gitignore).

## 2. About the repo

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
  evals/         post-hoc evaluation over the test split
  utils/         logger, meters, small tensor/plot utilities
packages/
  mdsconverter/  raw forgery datasets -> MDS shards (see DATASET_PIPELINE.md)
configs/         one yml per experiment (normal/, pdf/) + dataset conversion (mds/)
sagemaker/       SageMaker training entry point (see sagemaker/README.md)
notebooks/       thin launcher scripts (config path in, pipeline out)
scripts/         install / artifact-download / setup
tests/           smoke tests + single-batch overfit sanity check
checkpoints/     training output — see "Checkpoint/output layout" below
```

Inside `authgenforge/networks/`:

| File | Exposes | What it is |
|---|---|---|
| `dinov3_segmentation.py` | `DINOv3Segmentation`, `build_dinov3_segmentation`, `load_checkpoint` | DINOv3 ViT-L/16 (frozen, LoRA on `q_proj`/`k_proj`/`v_proj`) + a 3-convolution head (1024 → 512 → 256 → 1), bilinear-upsampled back to input resolution |

Inside `authgenforge/losses/`:

| File | Exposes | Notes |
|---|---|---|
| `bce_loss.py` | `PixelBCEWithLogitsLoss` | plain per-pixel BCE |
| `edge_bce_loss.py` | `EdgeWeightedBCEWithLogitsLoss` | BCE reweighted toward forgery-boundary pixels |
| `forgery_loss.py` | `ForgerySegmentationLoss` | `pixel_bce + edge_lambda * edge_bce`, default `edge_lambda=20.0` |

Inside `authgenforge/data/`:

| File | Exposes | What it reads |
|---|---|---|
| `forensics_dataset.py` | `ForensicsDataset` | loose `images/` + `masks/` folders (`data_format: folder`) |
| `dataloader.py` | `build_dataloaders` | train/test loaders for the folder backend |
| `forensics_mds_dataset.py` | `ForensicsMDSDataset` | MDS shards from a local directory or an `s3://` path (`data_format: mds`) |
| `forensics_mds_dataloader.py` | `build_mds_dataloaders` | train/test loaders for the MDS backend |

Every dataset returns `{image, mask, edge_mask, label}` — `label` is 0 for
authentic, 1 for tampered.

Inside `authgenforge/options/`:

| File | Role |
|---|---|
| `load.py` | pipeline builder — dataloaders, model, criterion, optimizer, scheduler, trainer |
| `option_utils.py` | yml parsing + path resolution shared by the loader and the evaluator |

### The core pattern: `data_format` + `_DATASET_BACKENDS`

This project has a single fixed architecture (DINOv3 ViT-L/16 + LoRA, binary
segmentation), so there is one loader (`load.py`), not a `_chosen`-name
dispatch over networks and losses. Swapping the backbone or loss is a
deliberate architecture change, not a config flip.

The dataset is the extension point:

1. `authgenforge/data/` holds one dataset module + one dataloader builder per
   storage format.
2. Every builder takes the same kwargs (`train_dir`, `test_dir`, `crop_size`,
   `batch_size`, `num_workers`, ...) and returns `(train_loader, test_loader)`.
3. The yml picks one by name:
   ```yaml
   data_format: mds             # or folder
   ```
4. `load.py`'s `_DATASET_BACKENDS` dict maps the name to the builder:
   ```python
   _DATASET_BACKENDS = {
       "folder": build_dataloaders,
       "mds": build_mds_dataloaders,
   }
   ```

Training, evaluation and the tests never import a specific dataset class by
name — switching data is a one-line config change.

### Config schema

The one yml has this top-level shape:

```yaml
name: train_forensics_mds_v1
model: dinov3_forensics_lora
type: binary_segmentation
data_context: normal          # normal | pdf — which configs/ subfolder this belongs to
data_format: mds              # mds | folder — see "The core pattern"
cache_dir: ...                # mds only: local cache when a dataroot is s3://
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

pretraining_settings:           # optional: load a previously trained Authenta checkpoint
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
  save_checkpoint_folder_path, load_checkpoint_file_path

eval_settings:                   # consumed by authgenforge/evals/evaluator.py
  checkpoint_path, image_size, threshold, batch_size, num_workers
  half_precision, output_dir
```

Two things worth knowing about how this gets parsed
(`authgenforge/options/option_utils.py`'s `parse_yml`):

- **Every path resolves relative to the yml file itself**, not the
  launching process's working directory. That covers `dataroot`,
  `structure.backbone.model_path`, `pretraining_settings.checkpoint_path`,
  `train_settings.{save_checkpoint_folder_path,load_checkpoint_file_path}`,
  and `eval_settings.{checkpoint_path,output_dir}`. URLs such as
  `s3://...` are left unchanged.
- **Missing keys return `None`, not `KeyError`** (`NoneDict`). `load.py`'s
  local `_cfg(value, default)` then falls back to a default only when a
  value is genuinely unset, so an explicit `0` or `False` in the yml is
  respected, not overridden.

### Checkpoint/output layout

```
checkpoints/{experiment_name}/
  latest_checkpoint.pth       full state: model, optimizer, scheduler, scaler,
                               epoch, global_step, best_iou
  {experiment_name}_best.pth  copy of latest_checkpoint.pth at the best val IoU so far
  epoch{N}.pth                 end-of-epoch, model-only snapshot
  {run_tag}/                   per-run-day logs/plots/predictions
    logs/log.txt
    plots/
    predictions/
  eval/                        authgenforge/evals/evaluator.py output
    predictions.csv
    metrics.json
```

## 3. Train, evaluate, and play around

Two existing configs to start from:

| Config | Trains on | Launch with |
|---|---|---|
| `configs/normal/train_forensics_mds.yml` | processed-v1 MDS shards (`data_format: mds`) — 1,827,437 train / 27,604 test images, see [DATASETS.md](DATASETS.md) | `notebooks/train.py --config ...` |
| `configs/normal/train_forensics.yml` | loose `images/` + `masks/` folders (`data_format: folder`) | `notebooks/train.py` (default config) |

### Train

Before running, check these fields in the config:

- `datasets.train.dataroot` / `datasets.test.dataroot` — your data. For
  `mds`, an MDS split directory (`<out>/train`, `<out>/test`, each holding
  `index.json` + `shard.*.mds`) or the same split's `s3://` path with
  `cache_dir` set. For `folder`, a directory with `images/` and `masks/`.
  A list of directories also works.
- `structure.backbone.model_path` — the local DINOv3 ViT-L/16 directory
  (see [Installation](#1-installation)).
- `datasets.train.batch_size` — the most common thing to change for GPU
  memory, offset by `train_settings.grad_accum_steps` for a larger
  effective batch size.
- `datasets.train.crop_size` — the training crop (default 512). Stored
  images are full size, so any crop works without rebuilding the data.
- `epoch_settings.total_epochs` — how long to train.

Then run:

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

Checkpoints and logs land under `checkpoints/{name}/` — `trainer.ckpt_dir`
and `trainer.log_dir` print the exact paths at startup. See
[Checkpoint/output layout](#checkpointoutput-layout).

### Resume a run

Set `train_settings.load_checkpoint_file_path` to a checkpoint file and
re-launch the same config. The path resolves relative to the yml.

The model weights, optimizer, scheduler, scaler, `epoch`, `global_step` and
`best_iou` are restored. The dataloader position is not: a resumed run starts
its epoch from a fresh shuffle.

To start a new run from a previous run's weights only, use
`pretraining_settings.{want_load, checkpoint_path}` instead.

### Evaluate

Before running, check these fields in `eval_settings`:

- `checkpoint_path` — which trained checkpoint to score. Typically the
  `{experiment_name}_best.pth` written during training.
- `image_size` — the center-crop size the test images are evaluated at.
- `threshold` — probability above which a pixel counts as tampered.
- `batch_size`, `num_workers`, `half_precision` — speed/memory.
- `output_dir` — where `predictions.csv` and `metrics.json` land (default
  `checkpoints/{name}/eval/`).

```python
from authgenforge.evals.evaluator import evaluate_from_yml
metrics = evaluate_from_yml("configs/normal/train_forensics_mds.yml")
```

Or `python notebooks/eval.py --config configs/normal/train_forensics_mds.yml`.

**What this does, step by step:**

1. Builds the test split from `datasets.test` with the same `data_format`
   switch training uses, center-cropped to `image_size`.
2. Loads `checkpoint_path` into the model and runs it over every batch under
   `torch.inference_mode()` (half precision on GPU if enabled).
3. Thresholds the predicted mask and accumulates pixel-level TP/TN/FP/FN
   across the whole split — no predicted masks are kept in memory.
4. Reports IoU, F1, precision, recall and accuracy, and writes them to
   `metrics.json`.

### Play around / debug

Fast ways to check something works without waiting on a real training run:

**Smoke test** — a few real training steps end to end (imports, data
loading, forward/backward, checkpoint save, inference), no full epoch:
```bash
python tests/smoke_test_pipeline.py --config configs/normal/train_forensics_mds.yml           # CUDA
python tests/smoke_test_pipeline.py --config configs/normal/train_forensics_mds.yml --cpu     # CPU
```

**Overfit one batch** — the model should drive loss toward zero within a
few hundred iterations:
```bash
python tests/overfit_single_batch.py --config configs/normal/train_forensics_mds.yml
```
If it can't overfit one batch, that's an architecture/loss wiring bug —
check this before spending GPU time debugging a full run that isn't
converging.

**Check the data** — no model weights needed:
```bash
python tests/smoke_test_mds.py                      # the converter, on a tiny generated dataset
python tests/smoke_test_all_paths.py --skip-s3      # every way the processed dataset loads (local)
python tests/smoke_test_s3.py                       # the processed dataset loaded from S3 (needs S3 read access)
```

**Poke one piece in isolation** — every loader function is a plain Python
function taking a yml path:
```python
from authgenforge.options.load import (
    get_model_from_yml,
    get_criterion_from_yml,
    get_dataloaders_from_yml,
)

model     = get_model_from_yml("configs/normal/train_forensics_mds.yml")
criterion = get_criterion_from_yml("configs/normal/train_forensics_mds.yml")
train_loader, test_loader = get_dataloaders_from_yml("configs/normal/train_forensics_mds.yml")
```

## 4. Contributing

### Add a new dataset backend

Nothing in `authgenforge/options/`, `authgenforge/training/`, or
`authgenforge/evals/` needs to change beyond one registry entry:

1. Add `authgenforge/data/{name}_dataset.py` — a `Dataset` returning
   `{image, mask, edge_mask, label}` — and `{name}_dataloader.py` with a
   `build_..._dataloaders(...)` builder matching the kwargs of
   `build_dataloaders` / `build_mds_dataloaders`.
2. Register it in `load.py`'s `_DATASET_BACKENDS` dict and handle it in the
   evaluator's `_build_test_dataset`.
3. Copy `configs/normal/train_forensics_mds.yml`, set `data_format` to the
   new name and point `datasets.train/test.dataroot` at your data.

### Add or rebuild a dataset in the MDS format

`packages/mdsconverter/` turns one folder per source dataset
(`images/authentic/`, `images/tampered/`, `masks/tampered/`) into MDS shards
per split. Add the dataset to `configs/mds/datasets.yml`, then follow
[DATASET_PIPELINE.md](DATASET_PIPELINE.md) (validate → remediate → convert)
and [MDS_DATASET.md](MDS_DATASET.md) (format, columns, verification).
Record the new counts in [DATASETS.md](DATASETS.md).

### Before opening a PR

- Run `tests/smoke_test_pipeline.py`. It catches import/wiring breaks
  across the whole pipeline in a couple minutes, not a couple hours.
- If you touched the model or a loss, run `tests/overfit_single_batch.py`
  — it should still drive loss to (near-)zero on one frozen batch. If it
  can't anymore, you've broken something structural, not just changed a
  metric.
- If you touched `authgenforge/data/` or `packages/mdsconverter/`, run
  `tests/smoke_test_mds.py` and `tests/smoke_test_all_paths.py`.
- If you add a new config key that holds a file path, resolve it in
  `option_utils.py`'s `parse_yml` (add it to the relevant path-resolution
  block) rather than resolving it ad hoc at the call site — otherwise it
  only works by coincidence, depending on the launching process's working
  directory.

### Conventions

- **No hardcoded per-backend dispatch** in shared code (`options/`,
  `training/`, `evals/`) beyond the `data_format` registry — a new data
  format is a new module plus one entry, not an `if` scattered through the
  trainer.
- **Keep diffs minimal.** This codebase favors small, focused modules over
  premature abstraction.
- **Comments explain why, not what.** A hidden constraint, a workaround
  for a specific bug, a version-pin reason. Self-explanatory code doesn't
  get a comment.
