# general-image-tampering-detecction

A config-driven training framework for the Authenta general image forgery
segmentation model.

It ships with a DINOv3 ViT-L/16 + LoRA backbone and a lightweight
convolutional segmentation head, predicting a dense per-pixel binary
forgery mask (`B x 1 x H x W`) rather than an image-level real/fake label.

## Contents

- [1. Installation](#1-installation)
- [2. About the repo](#2-about-the-repo)
- [3. Train, evaluate, and play around](#3-train-evaluate-and-play-around)
- [4. SageMaker training](#4-sagemaker-training)
- [5. Contributing](#5-contributing)

## 1. Installation

```bash
bash scripts/install_deps.sh       # installs deps with every version pin that matters
bash scripts/download_artifacts.sh  # fetches the DINOv3 ViT-L/16 backbone
```

Or run `bash scripts/setup.sh`, which does both in one shot for a fresh
session.

`install_deps.sh` pins a few versions for non-obvious reasons:

| Pin | Why |
|---|---|
| `numpy==1.26.4` | Lambda Stack's torch/torchvision are built against the NumPy 1.x ABI. NumPy 2.x breaks `torch.from_numpy()`/`.numpy()`. |
| `opencv-python-headless==4.11.0.86` | Kept below the release that forces `numpy>=2`. |
| `transformers==4.57.1` / `peft==0.19.1` | The exact pair the DINOv3 + LoRA loading path (`AutoModel.from_pretrained(..., local_files_only=True)` + `get_peft_model`) is verified against. A newer `peft` can require `transformers` internals (e.g. `HybridCache`) that an older pinned `transformers` doesn't export yet — install the two together, not independently. |

`download_artifacts.sh` fetches the official Hugging Face DINOv3 ViT-L/16
model (`facebook/dinov3-vitl16-pretrain-lvd1689m`) into
`artifacts/mirror/dinov3-vitl16/` — `config.json` + `model.safetensors`.
The weights are intentionally **not** committed to Git; see
[`.gitignore`](.gitignore).

## 2. About the repo

### Repo layout

```
authgenforge/
  networks/     DINOv3 ViT-L/16 + LoRA + segmentation head
  losses/       pixel BCE + edge-weighted BCE (forgery segmentation loss)
  data/         dataset + dataloader builders (Parquet — pending, see below)
  augmentations/ PIL/OpenCV-based paired image+mask augmentation pipelines
  optimizers/   optimizer (layer-decay AdamW) + LR scheduler
  options/      yml -> pipeline builders
  training/     SegmentationTrainer (train/validate/checkpoint/resume)
  evals/        post-hoc evaluation over a held-out dataset
  utils/        logger, meters, small tensor/plot utilities
configs/        one yml per experiment, grouped by data domain (normal/, pdf/)
sagemaker/      SageMaker-specific training entry point (see section 4)
notebooks/      thin launcher scripts (config path in, pipeline out)
scripts/        install / artifact-download / setup
tests/          smoke test + single-batch overfit sanity check
checkpoints/    training output — see "Checkpoint/output layout" below
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

Inside `authgenforge/options/`:

| File | Role |
|---|---|
| `load.py` | pipeline builder — dataloaders, model, criterion, optimizer, scheduler, trainer |
| `option_utils.py` | yml parsing + path resolution shared by the loader and the evaluator |

Unlike a multi-phase / multi-backbone framework, this project has a single
fixed architecture (DINOv3 ViT-L/16 + LoRA, binary segmentation), so there
is one loader (`load.py`), not a `_chosen`-name dynamic-dispatch registry.
Swapping the backbone or loss is a deliberate architecture change, not a
config flip.

### Config schema

The one yml has this top-level shape:

```yaml
name: train_forensics_v1
model: dinov3_forensics_lora
type: binary_segmentation
data_context: normal          # normal | pdf — which configs/ subfolder this belongs to

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
  save_checkpoint_folder_path, load_checkpoint_file_path, resume_dataloader

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
  and `eval_settings.{checkpoint_path,output_dir}`.
- **Missing keys return `None`, not `KeyError`** (`NoneDict`). `load.py`'s
  local `_cfg(value, default)` then falls back to a default only when a
  value is genuinely unset, so an explicit `0` or `False` in the yml is
  respected, not overridden.

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

## 3. Train, evaluate, and play around

Before running, check these fields in
[`configs/normal/train_forensics.yml`](configs/normal/train_forensics.yml):

- `structure.backbone.model_path` — the local DINOv3 ViT-L/16 directory
  (see [Installation](#1-installation)).
- `datasets.train.dataroot` / `datasets.test.dataroot` — your data.
- `datasets.train.batch_size` — the most common thing to change for GPU
  memory, offset by `train_settings.grad_accum_steps` for a larger
  effective batch size.
- `epoch_settings.total_epochs` — how long to train.

> The dataset implementation itself
> (`authgenforge/data/forensics_dataset.py`) is intentionally deferred
> until the training data's Parquet schema is finalized — see
> [`sagemaker/README.md`](sagemaker/README.md#current-status). Everything
> above it in the pipeline (config parsing, model/criterion/optimizer/
> trainer construction) is wired and testable today.

### Train

```bash
python notebooks/train.py --config configs/normal/train_forensics.yml --end_epoch 10
```

or inline:

```python
from authgenforge.options.load import load_pipeline_from_yml

train_loader, test_loader, model, trainer = load_pipeline_from_yml(
    "configs/normal/train_forensics.yml"
)
trainer.train_model(end_epoch=10)
```

Checkpoints and logs land under `checkpoints/{name}/` — `trainer.ckpt_dir`
and `trainer.log_dir` print the exact paths at startup. See
[Checkpoint/output layout](#checkpointoutput-layout).

### Resume a run

Set `train_settings.load_checkpoint_file_path` to a checkpoint file and
re-launch the same config. The path resolves relative to the yml.
`train_settings.resume_dataloader: true` (with `datasets.train.stateful_loader: true`)
picks up at the exact dataloader shard/shuffle position instead of
reshuffling from the top of the epoch.

### Evaluate

```bash
python notebooks/eval.py --config configs/normal/train_forensics.yml
```

or inline:

```python
from authgenforge.evals.evaluator import evaluate_from_yml
metrics = evaluate_from_yml("configs/normal/train_forensics.yml")
```

Reads `eval_settings.checkpoint_path` (required — typically the
`{experiment_name}_best.pth` written during training), runs the model over
`datasets.test` at a fixed threshold, and reports pixel-level TP/TN/FP/FN,
IoU, F1, precision, recall, and accuracy.

### Play around / debug

**Smoke test** — a few real training steps end to end (imports, data
loading, forward/backward, checkpoint save, inference), no full epoch:
```bash
python tests/smoke_test_pipeline.py           # CUDA
python tests/smoke_test_pipeline.py --cpu     # CPU
```

**Overfit one batch** — the model should drive loss toward zero within a
few hundred iterations. If it can't, that's an architecture/loss wiring
bug — check this before spending GPU time debugging a full run that isn't
converging:
```bash
python tests/overfit_single_batch.py
```

Both need `authgenforge/data/forensics_dataset.py` to exist (see the note
in [Train](#train) above) — until then they fail at the same, single,
expected point (`ModuleNotFoundError: authgenforge.data.forensics_dataset`).

## 4. SageMaker training

For running as a managed SageMaker Training job (S3 data, spot-interruptible,
checkpoints synced to S3) instead of a bare GPU box, see
[`sagemaker/README.md`](sagemaker/README.md). The canonical training
implementation stays under `authgenforge/`; `sagemaker/` is an
environment-specific addition, not a separate framework.

## 5. Contributing

Before opening a PR:

- Run `tests/smoke_test_pipeline.py`. It catches import/wiring breaks
  across the whole pipeline in a couple minutes, not a couple hours.
- If you touched the model or a loss, run `tests/overfit_single_batch.py`
  — it should still drive loss to (near-)zero on one frozen batch. If it
  can't anymore, you've broken something structural, not just changed a
  metric.
- If you add a new config key that holds a file path, resolve it in
  `option_utils.py`'s `parse_yml` (add it to the relevant path-resolution
  block) rather than resolving it ad hoc at the call site — otherwise it
  only works by coincidence, depending on the launching process's working
  directory.

### Conventions

- **Keep diffs minimal.** This codebase favors small, focused modules over
  premature abstraction.
- **Comments explain why, not what.** A hidden constraint, a workaround
  for a specific bug, a version-pin reason. Self-explanatory code doesn't
  get a comment.
- **Don't invent the data layer ahead of the schema.** `authgenforge/data/`
  and the Parquet format it will read are intentionally deferred — see
  `sagemaker/README.md`. Build against it once it lands, not before.
