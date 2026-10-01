# Training pipeline (SageMaker, automated)

Click a button in GitHub → a GPU in AWS trains the forgery-segmentation model
on the MDS data in S3 → checkpoints land back in S3. No laptop, no access keys.

Region **us-east-1** (where `authenta-data-rnd` lives — SageMaker input
channels must be in the job's region) · instance **ml.g6e.2xlarge** (1× L40S
48 GB) · on-demand.

---

## 1. The big picture

```
 GitHub Actions                          AWS (us-east-1)
 ──────────────                          ───────────────
 Build training image ──docker push───►  ECR  forgery-train:<sha>
 Deploy pipelines     ──create/update─►  Step Functions  forgery-training, forgery-smoke
 Smoke test / Run training ──start────►  Step Functions
                                               │ CreateTrainingJob (.sync — waits)
                                               ▼
                                         SageMaker
                                           1. starts a GPU instance
                                           2. pulls the image from ECR
                                           3. mounts S3 → /opt/ml/input/data/{train,test,artifacts}/
                                           4. runs  python sagemaker/train.py --config <yml> --end-epoch N
                                           5. syncs /opt/ml/checkpoints/ ↔ S3

 Pipeline checks (every push/PR, no AWS): static agreement tests +
 the real image built on a CPU base and run SageMaker-style.
```

**Our code never pulls images or mounts S3.** The state machine *describes*
the job (image, data, instance); SageMaker does the work.

## 2. Terms

| term | meaning |
|---|---|
| **ECR** | AWS's Docker registry. Stores the training image. |
| **Image tag** | version label of an image. Every build is tagged with its git commit SHA (+ `latest` on the default branch). |
| **Step Functions / state machine** | AWS workflow service. Ours has one step: "create a training job and wait for it". |
| **ASL** | the JSON language state machines are written in (`*.asl.json`). |
| **Channel** | one S3 prefix given to the job. Channel `train` appears in the container at `/opt/ml/input/data/train/`. |
| **FastFile** | channel mode that streams S3 objects on read — no upfront copy of the 1.1 TB train split. |
| **MDS** | MosaicML Streaming shards (`index.json` + `shard.*.mds`) — the processed-v1 format. Read in place from the FastFile mount. |
| **OIDC** | lets GitHub log into AWS with a short-lived token instead of stored access keys. |

## 3. Files

| file | what it is |
|---|---|
| `sagemaker/train.py` | training entry point (`--config`, `--end-epoch`). Detects a restored checkpoint and resumes. |
| `sagemaker/config/normal/train_forensics.yml` | real training config (hyperparameters + `/opt/ml/...` paths) |
| `sagemaker/config/smoke.yml` | tiny config: smoke slice, 1 epoch, batch 2 |
| `sagemaker/backbone/dinov3-vitl16/config.json` | DINOv3 ViT-L/16 config (the weights come from S3) |
| `sagemaker/requirements.txt` | Python deps installed into the image |
| `sagemaker/Dockerfile` | the image: AWS PyTorch 2.3 DLC + `authgenforge/` + `sagemaker/` |
| `.dockerignore` | keeps weights, data and unrelated folders out of the image |
| `infra/train-pipeline.asl.json` | state machine **`forgery-training`** — the real run |
| `infra/smoke-pipeline.asl.json` | state machine **`forgery-smoke`** — the smoke test |
| `infra/iam/*.json` | the three roles' trust + permission policies |
| `infra/render.sh` | fills `${AWS_ACCOUNT_ID}` / `${AWS_REGION}` into a template |
| `infra/setup-aws.sh` | one-time AWS setup (roles, OIDC, ECR) |
| `scripts/make_smoke_mds.py` | builds + uploads the smoke slice of processed-v1 |
| `tests/sagemaker/` | static agreement tests, SageMaker-style container test, local emulator |
| `.github/workflows/build-image.yml` | **Build training image** |
| `.github/workflows/deploy-pipeline.yml` | **Deploy pipelines** |
| `.github/workflows/smoke-test.yml` | **Smoke test** |
| `.github/workflows/run-training.yml` | **Run training** |
| `.github/workflows/pipeline-checks.yml` | **Pipeline checks** — automatic, no AWS |
| `sagemaker/launch.py` | the manual way (SageMaker estimator from a laptop). Not used by the pipeline. |

Inside the image the layout mirrors the repo, so every relative path still works:

```
/opt/ml/code/
├── authgenforge/
└── sagemaker/   train.py  config/  backbone/  requirements.txt
```

## 4. S3 layout

```
s3://authenta-data-rnd/image-tampering-detection/
├── processed-v1/{train,test}/        the dataset (MDS)             channel train / test
├── smoke/{train,test}/               smoke slice (MDS)             smoke channels
├── artifacts/dinov3-vitl16/          config.json + model.safetensors   channel artifacts
└── sagemaker/                        everything the jobs write
    ├── checkpoints/<job>/<experiment>/   latest_checkpoint.pth  <experiment>_best.pth  epochN.pth
    ├── output/<job>/
    └── smoke/{checkpoints,output}/<job>/
```

The SageMaker role can read `image-tampering-detection/*` but write only
`image-tampering-detection/sagemaker/*` — a job can never overwrite the dataset.

## 5. One-time setup

### AWS (admin, account `200283853008`)

```bash
bash infra/setup-aws.sh
```

Creates, idempotently:

| # | what | why |
|---|---|---|
| 1 | GitHub OIDC provider | Actions logs in without access keys |
| 2 | role `forgery-github-actions-role` | Actions: push to ECR, deploy + start state machines, read job results |
| 3 | role `forgery-stepfunctions-role` | Step Functions: create/describe training jobs |
| 4 | role `forgery-sagemaker-role` | the job: pull the image, read the data, write `sagemaker/*` |
| 5 | ECR repository `forgery-train` | |

Also check **Service Quotas → SageMaker → `ml.g6e.2xlarge for training job usage` ≥ 1**
in us-east-1 (it is 0 on many accounts; the job fails with `ResourceLimitExceeded`).
If the bucket uses SSE-KMS with a customer key, add `kms:Decrypt` (and
`kms:GenerateDataKey` for writes) on that key to `infra/iam/sagemaker-policy.json`.

```
GitHub ──OIDC──► forgery-github-actions-role   push to ECR, deploy + start state machines
                        │ passes
Step Functions ──► forgery-stepfunctions-role  create training jobs
                        │ passes
SageMaker job ──► forgery-sagemaker-role       pull image, read data, write checkpoints
```

### GitHub

Repo → Settings → Secrets and variables → Actions → New repository secret

- Name: `AWS_ROLE_ARN`
- Value: `arn:aws:iam::200283853008:role/forgery-github-actions-role`

That's the only secret.

### Data (once)

```bash
# DINOv3 ViT-L/16 backbone — gated on Hugging Face, needs an account with access
bash scripts/download_artifacts.sh
aws s3 cp --recursive artifacts/mirror/dinov3-vitl16/ \
    s3://authenta-data-rnd/image-tampering-detection/artifacts/dinov3-vitl16/ \
    --exclude '*' --include config.json --include model.safetensors

# smoke slice: 64 train + 32 test samples copied from processed-v1 (downloads ~1 GB)
python scripts/make_smoke_mds.py --upload
```

## 6. Running it

All AWS workflows are **manual**: Actions tab → pick one → **Run workflow**.

**First time, in this order:**

1. **Build training image** — ~15–20 min. The summary shows the image URI and its SHA tag.
2. **Deploy pipelines** — creates both state machines.
3. **Smoke test** — ~10–15 min. Waits and goes ✅/❌, with metrics, failure reason and
   the last log lines in the summary. Don't start a real run until it's green.
4. **Run training** — run name, SHA tag, epochs, time limit. Returns immediately.

**After that, only rerun what your change needs:**

| you changed | Build image | Deploy pipelines | Smoke test | Run training |
|---|:-:|:-:|:-:|:-:|
| nothing — just train again | | | | ✓ |
| run name / epochs / time limit | | | | ✓ |
| training code (`authgenforge/`, `train.py`) | ✓ | | ✓ | ✓ |
| a hyperparameter in `sagemaker/config/*.yml` | ✓ | | ✓ | ✓ |
| `requirements.txt` / `Dockerfile` | ✓ | | ✓ | ✓ |
| dataset S3 path, instance, volume | | ✓ | | ✓ |

## 7. Where to make changes

| I want to… | edit | then run |
|---|---|---|
| change a hyperparameter | `sagemaker/config/normal/train_forensics.yml` | Build → Smoke → Run |
| change training code | `authgenforge/…` | Build → Smoke → Run |
| add a Python package | `sagemaker/requirements.txt` | Build → Smoke → Run |
| point at a new dataset | `S3Uri` in `infra/train-pipeline.asl.json` **and** `sagemaker.inputs` in the yml | Deploy → Run |
| change instance / disk | `ResourceConfig` in the ASL (both files if it applies to both) | Deploy → Run |
| switch real runs to spot | add `"EnableManagedSpotTraining": true` + `"MaxWaitTimeInSeconds"` (≥ max runtime) to `train-pipeline.asl.json`; `train.py` already resumes from the restored checkpoint | Deploy → Run |
| add a second training config | new yml under `sagemaker/config/`, add it to `options:` in `run-training.yml` | Build → Run |

The **Pipeline checks** workflow fails if these drift apart: a channel without a
matching `dataroot`, a `sagemaker.inputs` URI that differs from the ASL, an S3
path the SageMaker role can't read/write, a metric regex that no longer matches
the trainer's log line, or mismatched role/repo/state-machine names.

⚠️ **Config lives in the image.** Editing a yml does nothing until you rebuild.

## 8. Smoke vs real — same image, different job

| | `forgery-smoke` | `forgery-training` |
|---|---|---|
| `--config` | `config/smoke.yml` (fixed) | chosen in the Run training form |
| data | `smoke/{train,test}` (96 samples) | `processed-v1/{train,test}` (1.83 M / 27.6 k samples) |
| backbone | `artifacts/dinov3-vitl16` | same |
| time limit / disk | 1 h / 50 GB | form input (default 72 h) / 200 GB |
| output | `…/sagemaker/smoke/…` | `…/sagemaker/…` |
| workflow | waits for the result | fires and returns |

One epoch of processed-v1 is ~457 k batches at batch 4 (~57 k optimizer steps
with grad-accum 8) — size `max_runtime_hours` for the epochs you ask for, or
the job is stopped mid-run
(the last `latest_checkpoint.pth` is in S3; start a new run from it via
`pretraining_settings.checkpoint_path`).

## 9. Outputs and logs

```
checkpoints  s3://authenta-data-rnd/image-tampering-detection/sagemaker/checkpoints/<job>/<experiment>/
logs         CloudWatch → /aws/sagemaker/TrainingJobs/<job>
metrics      SageMaker → Training jobs → <job> → Metrics
             train:{loss,iou,f1}  val:{loss,iou,f1,precision,recall,accuracy}
provenance   first lines of the log: [train] git_sha=<commit> image=<ecr uri>
```

The run name becomes the job name and the S3 folder; the workflow appends a
UTC timestamp because job names can't be reused. **Use the SHA tag for real
runs, not `latest`** — that's how you know which code made which checkpoint.

## 10. Testing without AWS

```bash
pytest tests/sagemaker -q                         # agreement tests, seconds
bash tests/sagemaker/container_test.sh            # needs Docker; ~15 min on CPU
python tests/sagemaker/emulate_training_job.py prepare --root /tmp/smjob
python tests/sagemaker/emulate_training_job.py local   --root /tmp/smjob --crop-size 224   # no Docker
```

`container_test.sh` builds `sagemaker/Dockerfile` on a CPU stand-in for the DLC,
then runs the image exactly as the smoke state machine does — same entrypoint,
arguments and baked-in `smoke.yml`, channels mounted read-only — on synthetic
MDS data and a random-init backbone. It checks the exit code, the checkpoints,
that every ASL metric regex matches the log, and that a second launch resumes
from the first one's checkpoint (a spot restart).

## 11. When it fails

| fails at | usually means |
|---|---|
| "Configure AWS credentials" | wrong `AWS_ROLE_ARN` secret, or the trust policy doesn't name this repo |
| Build: `FROM …763104351884…` | base image unreachable / wrong tag |
| Deploy: `AccessDenied … PassRole` | GitHub role can't pass `forgery-stepfunctions-role` |
| Smoke: "missing …/smoke/…" or "…/artifacts/…" | data not uploaded (§5 Data) |
| execution fails within seconds | Step Functions role missing a permission (`infra/iam/stepfunctions-policy.json`) |
| job: `ResourceLimitExceeded` | GPU quota is 0 (§5) |
| job: can't pull image | SageMaker role has no ECR read on `forgery-train` |
| job: `AccessDenied` on S3 | SageMaker role policy, or a KMS key on the bucket (§5) |
| job: `failed to open MDS dataset` | channel prefix empty or missing `index.json` |
| job: Python traceback | real bug — the smoke summary shows the last log lines; full log in CloudWatch |
