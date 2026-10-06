# general-image-tampering-detecction

Trains the Authenta image-tampering model: given an image, it marks which
pixels were edited. The model is DINOv3 ViT-L/16 with LoRA and a small
segmentation head.

There are two ways to train it, each with its own guide:

| Where it trains | Guide |
|---|---|
| **This machine's GPU** | this file — follow the steps below from top to bottom |
| **AWS SageMaker** (a rented cloud GPU) | [sagemaker/README.md](sagemaker/README.md) |

---

# Train on your own machine

Follow the steps in order. Every step shows the command to type and what you
should see. If what you see is different, go to
[If something goes wrong](#if-something-goes-wrong).

**What you need before you start**

- A computer running Ubuntu 22.04 or 24.04 with an NVIDIA graphics card
  (RTX 20-series or newer).
- About 80 GB of free disk space.
- An internet connection.
- An AWS access key and secret key that can read the bucket
  `authenta-data-rnd`. Ask your team lead for them.
- Access to this GitHub repository.

All commands are typed in a terminal (press `Ctrl+Alt+T` to open one).

## Step 1 — Check the graphics card

```bash
nvidia-smi
```

You should see a table with the name of your graphics card, for example
`NVIDIA GeForce RTX 5070 Ti`.

If you see `NVIDIA-SMI has failed`, the graphics driver is not working. Fix it
with [the driver steps](#the-graphics-driver-is-not-working) before going on.

## Step 2 — Install Miniconda

Miniconda keeps this project's Python packages separate from the rest of the
computer. First check whether it is already installed:

```bash
conda --version
```

If that prints a version number, skip to Step 3. If it says
`command not found`, install it:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
```

Close the terminal and open a new one. `conda --version` should now print a
version number.

## Step 3 — Download the code

```bash
cd ~
git clone https://github.com/phospheneai/general-image-tampering-detecction.git
cd general-image-tampering-detecction
git checkout sagemaker-pipeline
```

(The `git checkout` line is needed only until this branch is merged into the
repository's main branch. If it prints an error saying the branch does not
exist, ignore it.)

Every command from here on is run from inside this folder.

## Step 4 — Install the Python packages

```bash
conda create -n forgery python=3.11 -y
conda activate forgery
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
bash scripts/install_deps.sh
```

This takes several minutes. It ends with
`Authenta dependencies installed successfully.`

Your terminal prompt now starts with `(forgery)`. **Every time you open a new
terminal, run these two lines first:**

```bash
cd ~/general-image-tampering-detecction
conda activate forgery
```

## Step 5 — Connect to AWS

The training data and the starting model are stored in AWS. First install the
AWS command-line tool (skip this block if `aws --version` already prints a
version):

```bash
sudo apt-get update && sudo apt-get install -y unzip curl
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip -q awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip
```

Then enter your keys:

```bash
aws configure
```

It asks four questions. Answer them like this:

| Question | What to type |
|---|---|
| AWS Access Key ID | your access key |
| AWS Secret Access Key | your secret key |
| Default region name | `us-east-1` |
| Default output format | `json` |

Check that it works:

```bash
aws s3 ls s3://authenta-data-rnd/image-tampering-detection/
```

You should see a short list that includes `artifacts/`, `processed-v1/` and
`smoke/`.

## Step 6 — Install the tool that mounts the dataset

The dataset is 1.1 TB, too large to download. Instead it is *mounted*: it
appears as a normal folder on your computer, and files are fetched from AWS
when they are read. This tool does the mounting:

```bash
bash scripts/install_mountpoint.sh
```

It asks for your computer password. It ends by printing a version, for example
`[install] mount-s3 1.24.0`.

You do not need to mount anything yourself. The training command in the next
step does it.

## Step 7 — Run the quick test

This trains on 64 images for one round. It proves that everything is set up
correctly before you start a long run.

```bash
python train.py --smoke
```

You should see lines like these, in this order:

```
[train] GPU NVIDIA GeForce RTX 5070 Ti ...
[train] AWS arn:aws:iam::...
[train] dataset s3://authenta-data-rnd/image-tampering-detection/ mounted at /home/<you>/data/s3/image-tampering-detection
[train] backbone .../artifacts/mirror/dinov3-vitl16
... [Train] Epoch 1 | loss ... | IoU ... | F1 ...
... [Val] Epoch 1 | loss ... | IoU ... | F1 ...
[train] done — best checkpoint .../checkpoints/local_smoke/local_smoke_best.pth
```

The first time, it also downloads the starting model (about 1.2 GB), so it
takes longer. On our test machine the quick test took about 6 minutes; most
of that is reading images from AWS, so it depends on your connection.

When you see `[train] done`, the setup works.

If you run the quick test a second time it says `already trained to epoch 1 —
nothing to do`. To run it again from scratch, delete its results first:
`rm -rf checkpoints/local_smoke`.

## Step 8 — Run the real training

A real run takes days, so start it inside `tmux`. That keeps it running after
you close the terminal or lose your connection.

```bash
sudo apt-get install -y tmux
tmux new -s train
```

Inside the tmux window:

```bash
cd ~/general-image-tampering-detecction
conda activate forgery
python train.py
```

- To leave it running and close the window: press `Ctrl+B`, release, then press `D`.
- To look at it again later: `tmux attach -t train`.

By default it trains for 10 rounds (called epochs) over 1.83 million images.
To stop earlier, give the number of epochs:

```bash
python train.py --epochs 1
```

## Step 9 — Stop and continue

- **To stop:** press `Ctrl+C` in the window where training is running.
- **To continue:** run the same command again (`python train.py`). It finds
  the last saved state and carries on from there. This also works after a
  crash or a restart of the computer.

Progress is saved every 1,000 model updates (8,000 batches) and at the end of
every epoch, so stopping loses at most the work since the last save.

## Step 10 — Find the results

Everything is saved in the `checkpoints` folder:

```
checkpoints/train_forensics_local_v1/
├── train_forensics_local_v1_best.pth   the best model so far  ← the one to use
├── latest_checkpoint.pth               the most recent state, used to continue
├── epoch1.pth, epoch2.pth, …           the model at the end of each epoch
├── metrics.csv                         one row per epoch: loss, IoU, F1, precision, recall, accuracy
└── <date_time>/
    ├── logs/log.txt                    everything that was printed
    └── predictions/val_epoch_<N>.csv   one row per test image
```

The quick test from Step 7 saves the same files under
`checkpoints/local_smoke/`.

To look at the scores per epoch:

```bash
column -s, -t checkpoints/train_forensics_local_v1/metrics.csv
```

The per-image file has these columns:

| Column | Meaning |
|---|---|
| `image_name` | which image |
| `ground_truth` | the correct answer: `0` = not edited, `1` = edited |
| `probability` | how sure the model is that the image was edited (0 to 1) |
| `predicted_class` | the model's answer: `0` = not edited, `1` = edited |
| `iou` | how well the predicted edited area overlaps the real one (0 to 1; higher is better) |

## If something goes wrong

| What you see | What to do |
|---|---|
| `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver` | See [The graphics driver is not working](#the-graphics-driver-is-not-working) below. |
| `[train] ERROR: no usable GPU` | Read the `Fix:` line printed under it and run that command. |
| `... is not compatible with the current PyTorch installation` or `no kernel image is available` | The wrong PyTorch is installed. Run `conda activate forgery`, then the `pip install torch==2.7.1 ...` line from Step 4 again. |
| `[train] ERROR: mount-s3 is not installed` | Run Step 6. |
| `the data is on S3 but AWS credentials don't work` | Run `aws configure` again (Step 5) and check the keys. |
| `could not mount s3://...` with `Access Denied` | Your AWS keys cannot read the bucket. Ask your team lead for access to `authenta-data-rnd`. |
| `... is not empty — refusing to mount over it` | The mount folder has files in it. Empty or delete `~/data/s3/image-tampering-detection` and run again. |
| `CUDA out of memory` | The graphics card does not have enough memory. Open `configs/local/train_forensics.yml`, change `batch_size: 4` to `batch_size: 2` and `grad_accum_steps: 8` to `grad_accum_steps: 16`, then run again. |
| `conda: command not found` | Close the terminal, open a new one, and try again. If it still fails, redo Step 2. |
| `ModuleNotFoundError` | You are not in the project environment. Run `conda activate forgery`. |
| Training is very slow | Images are read from AWS in the `us-east-1` region. On a slow or distant connection this is the limit, not the graphics card. Training on AWS itself avoids it: see [sagemaker/README.md](sagemaker/README.md). |

### The graphics driver is not working

This usually happens after Ubuntu updates itself: the system gets a new
kernel, but the NVIDIA driver for that kernel is missing. Run:

```bash
sudo apt update
sudo apt install linux-modules-nvidia-580-open-generic-hwe-24.04
sudo reboot
```

After the computer restarts, `nvidia-smi` should show the table from Step 1.
(If your machine uses a different driver version, `train.py` prints the exact
command for it in its `Fix:` line.)

### Disconnecting the dataset folder

The mounted folder stays connected until the computer restarts. To disconnect
it by hand:

```bash
python -m authgenforge.utils.s3_mount --unmount ~/data/s3/image-tampering-detection
```

It is mounted again automatically the next time you run `python train.py`.

---

# Reference

Everything below is background for people changing the code. You do not need
it to train.

## Package versions

`install_deps.sh` pins a few versions for non-obvious reasons:

| Pin | Why |
|---|---|
| `numpy==1.26.4` | The system torch/torchvision builds are compiled against the NumPy 1.x ABI. NumPy 2.x breaks `torch.from_numpy()`/`.numpy()`. It also means Python ≤ 3.12. |
| `opencv-python-headless==4.11.0.86` | Kept below the release that forces `numpy>=2`. |
| `transformers==4.57.1` / `peft==0.19.1` | The exact pair the DINOv3 + LoRA loading path (`AutoModel.from_pretrained(..., local_files_only=True)` + `get_peft_model`) is verified against. A newer `peft` can require `transformers` internals (e.g. `HybridCache`) that an older pinned `transformers` doesn't export yet — install the two together, not independently. |
| `mosaicml-streaming==0.13.0` | Reads and writes the MDS shards. Needs `numpy<2.2`, compatible with the pin above. |

## About the repo

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
configs/         one yml per experiment + dataset conversion (mds/)
sagemaker/       SageMaker training image: entry point, configs, Dockerfile
infra/           Step Functions pipelines + IAM for SageMaker training (see sagemaker/README.md)
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
| `forensics_mds_dataset.py` | `ForensicsMDSDataset` | MDS shards from a directory — local disk or a mounted S3 prefix (`data_format: mds`) |
| `forensics_mds_dataloader.py` | `build_mds_dataloaders` | train/test loaders for the MDS backend |

Every dataset returns `{image, mask, edge_mask, label, name}` — `label` is 0
for authentic, 1 for tampered; `name` is the image's original path.

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
data_format: mds              # mds | folder — see "The core pattern"
s3_mount:                     # optional, train.py only: mount this S3 prefix as a folder
  uri: s3://bucket/prefix/
  mount_point: ~/data/s3/prefix

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
  metrics.csv                  one row per epoch: train_* and val_* loss, iou, f1,
                               precision, recall, accuracy (appended to on resume)
  {run_tag}/                   per-run-day logs/plots/predictions
    logs/log.txt
    plots/
    predictions/
      val_epoch_{N}.csv        one row per test image: image_name, ground_truth,
                               probability, predicted_class, iou
  eval/                        authgenforge/evals/evaluator.py output
    predictions.csv
    metrics.json
```

## Other ways to run

`python train.py` (the steps above) is the simplest way. This section is the
lower-level way: any config, through `notebooks/train.py` or Python.

Two existing configs to start from:

| Config | Trains on | Launch with |
|---|---|---|
| `configs/train_forensics_mds.yml` | processed-v1 MDS shards (`data_format: mds`) — 1,827,437 train / 27,604 test images, see [DATASETS.md](DATASETS.md) | `notebooks/train.py --config ...` |
| `configs/train_forensics.yml` | loose `images/` + `masks/` folders (`data_format: folder`) | `notebooks/train.py` (default config) |

### Train

Before running, check these fields in the config:

- `datasets.train.dataroot` / `datasets.test.dataroot` — your data. For
  `mds`, an MDS split directory (`<out>/train`, `<out>/test`, each holding
  `index.json` + `shard.*.mds`) — on local disk, or under a mounted S3
  prefix (`python -m authgenforge.utils.s3_mount <s3 uri> <folder>`). For `folder`, a directory with `images/` and `masks/`.
  A list of directories also works.
- `structure.backbone.model_path` — the local DINOv3 ViT-L/16 directory
  (`python train.py` downloads it; or `bash scripts/download_artifacts.sh`).
- `datasets.train.batch_size` — the most common thing to change for GPU
  memory, offset by `train_settings.grad_accum_steps` for a larger
  effective batch size.
- `datasets.train.crop_size` — the training crop (default 512). Stored
  images are full size, so any crop works without rebuilding the data.
- `epoch_settings.total_epochs` — how long to train.

Then run:

```bash
python notebooks/train.py --config configs/train_forensics_mds.yml --end_epoch 10
```

or inline:

```python
from authgenforge.options.load import load_pipeline_from_yml

train_loader, test_loader, model, trainer = load_pipeline_from_yml(
    "configs/train_forensics_mds.yml"
)
trainer.train_model(end_epoch=10)
```

Checkpoints and logs land under `checkpoints/{name}/` — `trainer.ckpt_dir`
and `trainer.log_dir` print the exact paths at startup. See
[Checkpoint/output layout](#checkpointoutput-layout) above.

### Resume a run

Set `train_settings.load_checkpoint_file_path` to a checkpoint file and
re-launch the same config. The path resolves relative to the yml.

The model weights, optimizer, scheduler, scaler, `epoch`, `global_step` and
`best_iou` are restored. With `stateful_loader: true` and
`resume_dataloader: true`, a mid-epoch checkpoint (every `save_interval`
optimizer updates) also restores the train loader's position, so the epoch
continues where it stopped instead of starting over. (`python train.py` and
the SageMaker job find `latest_checkpoint.pth` and set this up themselves.)

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
metrics = evaluate_from_yml("configs/train_forensics_mds.yml")
```

Or `python notebooks/eval.py --config configs/train_forensics_mds.yml`.

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
python tests/smoke_test_pipeline.py --config configs/train_forensics_mds.yml           # CUDA
python tests/smoke_test_pipeline.py --config configs/train_forensics_mds.yml --cpu     # CPU
```

**Overfit one batch** — the model should drive loss toward zero within a
few hundred iterations:
```bash
python tests/overfit_single_batch.py --config configs/train_forensics_mds.yml
```
If it can't overfit one batch, that's an architecture/loss wiring bug —
check this before spending GPU time debugging a full run that isn't
converging.

**Check the data** — no model weights needed:
```bash
python tests/smoke_test_mds.py                      # the converter, on a tiny generated dataset
python tests/smoke_test_all_paths.py --skip-s3      # every way the processed dataset loads (local)
python tests/smoke_test_s3.py                       # the processed dataset loaded from the mounted S3 prefix (needs S3 read access + mount-s3)
```

**Poke one piece in isolation** — every loader function is a plain Python
function taking a yml path:
```python
from authgenforge.options.load import (
    get_model_from_yml,
    get_criterion_from_yml,
    get_dataloaders_from_yml,
)

model     = get_model_from_yml("configs/train_forensics_mds.yml")
criterion = get_criterion_from_yml("configs/train_forensics_mds.yml")
train_loader, test_loader = get_dataloaders_from_yml("configs/train_forensics_mds.yml")
```

## Contributing

### Add a new dataset backend

Nothing in `authgenforge/options/`, `authgenforge/training/`, or
`authgenforge/evals/` needs to change beyond one registry entry:

1. Add `authgenforge/data/{name}_dataset.py` — a `Dataset` returning
   `{image, mask, edge_mask, label, name}` — and `{name}_dataloader.py` with a
   `build_..._dataloaders(...)` builder matching the kwargs of
   `build_dataloaders` / `build_mds_dataloaders`.
2. Register it in `load.py`'s `_DATASET_BACKENDS` dict and handle it in the
   evaluator's `_build_test_dataset`.
3. Copy `configs/train_forensics_mds.yml`, set `data_format` to the
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
- If you touched `sagemaker/`, `infra/`, or the trainer's log format, run
  `pytest tests/sagemaker` (CI also runs it, plus the container test).
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
