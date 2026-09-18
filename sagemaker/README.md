# SageMaker training

Run the Authenta general image forgery segmentation model as a SageMaker
Training job — training data from S3, checkpoints to S3, spot-interruptible.

The model is DINOv3 ViT-L/16 with LoRA and a dense binary segmentation head.

This `sagemaker/` directory contains the SageMaker-specific addition.
The canonical training implementation remains under `authgenforge/`.

## Files

| file | role |
| --- | --- |
| `config/normal/train_forensics.yml` | SageMaker training/environment config (`/opt/ml/...` paths) |
| `config/smoke.yml` | smoke-test configuration; dataset implementation is pending |
| `backbone/dinov3-vitl16/config.json` | DINOv3 ViT-L/16 configuration, ships with `source_dir` |
| `requirements.txt` | container dependencies, pinned to the 2.3/py311 DLC |
| `train.py` | SageMaker entry point — environment setup, config patching, pipeline construction, training |
| `launch.py` | launcher — reads the config's `sagemaker:` block and submits the job |

The root training configuration is separate:

```text
../configs/normal/train_forensics.yml
```

The SageMaker configuration is an environment-specific copy/adaptation.
Changes made to one configuration do not automatically propagate to the
other.

## Run it

Install the local SageMaker launcher dependencies:

```bash
pip install sagemaker pyyaml boto3
```

Set the SageMaker execution role:

```bash
export SAGEMAKER_ROLE=arn:aws:iam::<ACCT>:role/<ExecutionRole>
```

On Windows PowerShell:

```powershell
$env:SAGEMAKER_ROLE="arn:aws:iam::<ACCT>:role/<ExecutionRole>"
```

Before a real run, configure the following in:

```text
sagemaker/config/normal/train_forensics.yml
```

```yaml
sagemaker:
  region: <AWS_REGION>

  output_bucket: <S3_BUCKET>

  checkpoint_prefix: general-forensics/checkpoints

  output_prefix: general-forensics/output

  inputs:
    train:
      uri: <TRAIN_S3_URI>
      mode: FastFile

    test:
      uri: <TEST_S3_URI>
      mode: FastFile

    artifacts:
      uri: <ARTIFACTS_S3_URI>
      mode: File
```

Do not use the placeholder values literally.

The final dataset S3 locations and final Parquet layout are not yet
committed to this repository.

### Dry run

```bash
python sagemaker/launch.py --dry-run --run-name test-run
```

`--dry-run` validates the local SageMaker setup and prints the planned
configuration without submitting a training job.

### Full run

```bash
python sagemaker/launch.py --run-name <name>
```

The default configuration is:

```text
config/normal/train_forensics.yml
```

An explicit configuration can also be supplied:

```bash
python sagemaker/launch.py \
    --config config/normal/train_forensics.yml \
    --run-name <name>
```

### On-demand instead of Spot

Spot Training is enabled by default.

To use an on-demand instance:

```bash
python sagemaker/launch.py \
    --run-name <name> \
    --no-spot
```

### Submit without waiting for logs

```bash
python sagemaker/launch.py \
    --run-name <name> \
    --no-wait
```

The default instance is:

```text
ml.g6e.2xlarge
```

with:

```text
instance_count = 1
```

The trainer is single-process.

## How it works

- **Entry point** — `source_dir=sagemaker/` is shipped to the SageMaker
  container. The toolkit runs `python train.py` in script mode and installs
  `requirements.txt`. `authgenforge/` is outside `source_dir`, so the launcher
  supplies it through the estimator's `dependencies`.

- **Code layout** — the contents of `sagemaker/` are available under
  `/opt/ml/code/`. Therefore the SageMaker training configuration is available
  at `/opt/ml/code/config/normal/train_forensics.yml`, while the DINOv3
  configuration is available at
  `/opt/ml/code/backbone/dinov3-vitl16/config.json`.

- **Model** — the training pipeline uses DINOv3 ViT-L/16 with LoRA rank 32,
  alpha 64 and dropout 0.0, followed by a three-convolution segmentation head
  producing one forgery-logit channel.

- **Backbone configuration** — `backbone/dinov3-vitl16/config.json` is the
  ViT-L/16 configuration used by this project. The DINOv3 Huge configuration
  from the reference MIRROR repository is not used.

- **Input data** — the SageMaker configuration defines `train`, `test`, and
  `artifacts` input channels. SageMaker exposes them under:

  ```text
  /opt/ml/input/data/train
  /opt/ml/input/data/test
  /opt/ml/input/data/artifacts
  ```

  The final Parquet schema and `authgenforge/data/` implementation are
  intentionally deferred until the data format is finalized.

- **Artifacts** — the `artifacts` channel is intended to provide the base
  DINOv3 ViT-L/16 pretrained weights. The large pretrained weight file is not
  committed to the repository.

- **Run name** — `--run-name` is the SageMaker training job name and the S3
  sub-prefix for checkpoints/output. Runs therefore do not share checkpoint
  directories. SageMaker job names cannot be reused.

- **Checkpoints** — `/opt/ml/checkpoints` is synchronized with the run-specific
  S3 checkpoint prefix:

  ```text
  s3://<bucket>/general-forensics/checkpoints/<run-name>/
  ```

  The trainer writes `latest_checkpoint.pth` and the best/epoch checkpoints
  under the experiment checkpoint directory.

- **Spot resume** — when a Spot job is restarted and SageMaker restores
  `/opt/ml/checkpoints`, `train.py` detects the restored
  `latest_checkpoint.pth` and enables checkpoint loading and dataloader
  resume.

- **Initialization weights** — if an optional initialization checkpoint is
  configured but is not present in the `artifacts` channel, `train.py`
  disables `want_load` rather than failing during startup.

- **Single GPU** — the trainer is single-process, so `instance_count` is 1.

- **Monitoring** — SageMaker provides the training logs and configured
  metrics. The launcher disables the SageMaker profiler and debugger hooks.

## Configuration

There are two configuration layers.

### Canonical training configuration

```text
../configs/normal/train_forensics.yml
```

This defines the model and training behavior, including:

```text
model:
    dinov3_forensics_lora

type:
    binary_segmentation

backbone:
    dinov3_vitl16

LoRA:
    rank = 32
    alpha = 64
    dropout = 0.0
```

It also defines the optimizer, scheduler, loss, epochs and training
mechanics.

### SageMaker configuration

```text
config/normal/train_forensics.yml
```

This adapts the environment-specific paths and adds:

```yaml
sagemaker:
  region:
  output_bucket:
  checkpoint_prefix:
  output_prefix:
  inputs:
```

The two configuration layers should remain separate.

## Checkpoint and output layout

Each run receives its own S3 prefix.

Conceptually:

```text
s3://<bucket>/
├── general-forensics/
│   ├── checkpoints/
│   │   └── <run-name>/
│   │       └── <experiment-name>/
│   │           ├── latest_checkpoint.pth
│   │           ├── <experiment-name>_best.pth
│   │           └── epochN.pth
│   │
│   └── output/
│       └── <run-name>/
```

The exact files produced under the experiment directory are controlled by
the trainer.

The training output is not dependent on a `model.tar.gz` deployment artifact;
the `.pth` checkpoints are the relevant training artifacts.

## Local check

A complete local Parquet training check is intentionally deferred.

The final data implementation:

```text
authgenforge/data/
```

and the final Parquet schema have not yet been finalized.

Once the schema and dataset implementation are available, a local check can
be added using a small representative dataset and:

```bash
python sagemaker/train.py \
    --config <local-config> \
    --end-epoch 1
```

Until then, do not invent a Parquet schema or dataset loader solely for a
SageMaker smoke test.

## Current status

SageMaker infrastructure currently includes:

```text
sagemaker/
├── backbone/dinov3-vitl16/config.json
├── config/normal/train_forensics.yml
├── requirements.txt
├── train.py
├── launch.py
└── README.md
```

The following remain intentionally deferred:

```text
authgenforge/data/
final Parquet schema
final S3 dataset paths
final DINOv3 weight packaging
smoke-test execution
final SageMaker launch verification
```

These should be completed only after the corresponding data and artifact
locations are finalized.