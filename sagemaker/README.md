# SageMaker training

Run the Authenta general image forgery segmentation model (DINOv3 ViT-L/16 +
LoRA + dense segmentation head) as a SageMaker training job — MDS data from
S3, checkpoints to S3, restart-safe.

There are two ways to start a job; both run the same `train.py` with the same
configs:

| | how | use it for |
|---|---|---|
| **Pipeline** (default) | GitHub Actions → Step Functions → a custom ECR image | every real run — see [`../infra/README.md`](../infra/README.md) |
| **Launcher** | `python sagemaker/launch.py` → SageMaker PyTorch estimator | one-off runs from a laptop with AWS credentials |

## Files

| file | role |
| --- | --- |
| `train.py` | entry point — sys.path, restart detection, config patching, runs `SegmentationTrainer` |
| `config/normal/train_forensics.yml` | real run: processed-v1 MDS, `/opt/ml/...` paths, `sagemaker:` block for the launcher |
| `config/smoke.yml` | smoke test: smoke slice, 1 epoch, batch 2 |
| `backbone/dinov3-vitl16/config.json` | DINOv3 ViT-L/16 config (weights arrive on the `artifacts` channel) |
| `requirements.txt` | deps on top of the PyTorch 2.3 / py311 DLC |
| `Dockerfile` | the pipeline's image: DLC + `authgenforge/` + this directory |
| `launch.py` | the estimator launcher |

`config/` is a SageMaker-specific copy of the repo's `configs/` — hyperparameter
changes there do **not** propagate here.

## Container contract

| path in the container | from | config key |
|---|---|---|
| `/opt/ml/input/data/train/` | `s3://authenta-data-rnd/image-tampering-detection/processed-v1/train/` (FastFile) | `datasets.train.dataroot` |
| `/opt/ml/input/data/test/` | `…/processed-v1/test/` (FastFile) | `datasets.test.dataroot` |
| `/opt/ml/input/data/artifacts/` | `…/artifacts/dinov3-vitl16/` (File) | `structure.backbone.model_path` |
| `/opt/ml/checkpoints/` | synced both ways with `…/sagemaker/checkpoints/<job>/` | `train_settings.save_checkpoint_folder_path` |
| `/opt/ml/output/failure` | written by `train.py` on a crash → the job's FailureReason | |

The MDS shards are read in place from the FastFile mount (`data_format: mds`,
local `dataroot`) — no `cache_dir`, no upfront copy of the 1.1 TB train split.
Channel names and `dataroot`s must match; `pytest tests/sagemaker` fails if
they drift.

## How train.py behaves

- **Restart / spot resume** — if `<save_checkpoint_folder_path>/<name>/latest_checkpoint.pth`
  exists at startup (SageMaker restored it from S3), the effective config gets
  `load_checkpoint_file_path` = that file and `resume_dataloader: true`, and
  training continues from the saved epoch/step.
- **Init weights** — `pretraining_settings.want_load: true` with a
  `checkpoint_path` that isn't there is downgraded to `want_load: false`
  (backbone-only start) instead of failing.
- **Provenance** — logs `git_sha` (baked into the image) and the image URI.
- **Metrics** — the trainer prints `[Train|Val] Epoch N | loss … | IoU … | F1 … | precision … | recall … | accuracy …`;
  the regexes that scrape them live in `infra/*.asl.json` (`launch.py` reads
  the same list).
- Extra arguments SageMaker may append are ignored (`parse_known_args`).

## Launcher (manual path)

```bash
pip install sagemaker pyyaml boto3
export SAGEMAKER_ROLE=arn:aws:iam::<ACCT>:role/forgery-sagemaker-role

python sagemaker/launch.py --dry-run --run-name test-run      # print the plan, submit nothing
python sagemaker/launch.py --run-name <name>                  # spot by default
python sagemaker/launch.py --run-name <name> --no-spot --no-wait
```

Region, bucket and channels come from the config's `sagemaker:` block. The
estimator ships `sagemaker/` as `source_dir` (→ `/opt/ml/code/`) and
`authgenforge/` as a dependency; `train.py` handles both this layout and the
pipeline image's (`/opt/ml/code/sagemaker/`). Run names are job names and S3
sub-prefixes and cannot be reused.

## Testing

See [`../infra/README.md` §10](../infra/README.md#10-testing-without-aws):
`pytest tests/sagemaker`, `tests/sagemaker/container_test.sh` (the image,
SageMaker-style, on CPU), and `tests/sagemaker/emulate_training_job.py`
(no Docker).
