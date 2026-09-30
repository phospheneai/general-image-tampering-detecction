# processed-v1 — how to check it and load it in PyTorch

A step-by-step guide to confirm the processed image-tampering dataset is
complete and loads in PyTorch — from S3 or from the local copy — and how to use
it for training. Copy the commands exactly; each step shows what you should see.

## What it is

| | |
|---|---|
| **S3** | `s3://authenta-data-rnd/image-tampering-detection/processed-v1/` |
| **Local copy** (EC2 `ip-10-3-1-46`) | `/home/ubuntu/data/processed/processed-v1/` |
| **Format** | MosaicML Streaming (MDS): `train/` and `test/`, each an `index.json` + 512 MB `shard.*.mds` files |
| **train** | 1,827,437 images — 1,003,616 authentic / 823,821 tampered — 2,108 shards, 1,128.9 GB |
| **test** | 27,604 images — 6,329 authentic / 21,275 tampered — 28 shards, 14.6 GB |
| **Code** | repo `phospheneai/general-image-tampering-detecction`, branch `processed-v1-dataset` |

Every sample stores the **original image and mask bytes** (never re-encoded),
its **label** (0 = authentic, 1 = tampered), source dataset, original file path
and original size. Per-dataset counts: [DATASETS.md](DATASETS.md). How it was
built: [MDS_DATASET.md](MDS_DATASET.md), [DATASET_PIPELINE.md](DATASET_PIPELINE.md).
1,204 broken raw files were left out; they are listed in
`processed-v1/train_failed.csv`.

In PyTorch every sample (and batch) is:

| key | type / shape | meaning |
|---|---|---|
| `image` | float32 `(3, 512, 512)` | RGB, values 0–1 (512 = training crop; stored images are full size) |
| `mask` | float32 `(1, 512, 512)` | 1 = tampered pixel; all zero for authentic images |
| `edge_mask` | float32 `(1, 512, 512)` | band around the tampered region's boundary (used by the loss) |
| `label` | int64 `()` | **0 = authentic, 1 = tampered** |

A `DataLoader` batch adds the batch size in front: `(8, 3, 512, 512)`, …, `label (8,)`.

---

## Before you start: AWS access

Reading from S3 needs credentials with `s3:GetObject` on
`authenta-data-rnd/image-tampering-detection/processed-v1/*`.

> The EC2 instance's own role (`instanceRole-s3-access`) **cannot** read this
> bucket yet — use a personal profile. Without one you get
> `403 Forbidden` (or `index.json not found!`).

If you don't have a profile on the machine yet, create one (it asks for your
access key and secret — type them into the terminal, never paste them into chat
or commit them):

```bash
aws configure --profile <your-profile>
```

Check it works — this must print a file size, not `AccessDenied`/`403`:

```bash
aws s3 ls s3://authenta-data-rnd/image-tampering-detection/processed-v1/train/index.json --profile <your-profile>
```

---

## Option A — on the EC2 instance (everything already installed) · ~5 min

**A1. Open the project and the Python environment**

```bash
cd ~/general-image-tampering-detecction
git checkout processed-v1-dataset
git pull
source ~/venv/bin/activate
export AWS_PROFILE=<your-profile>
```

The prompt now starts with `(venv)`.

**A2. Load the real dataset from S3** (~30 s, downloads 1 GB into a temp folder, deleted afterwards)

```bash
python tests/smoke_test_s3.py
```

Expected (label lists vary slightly):

```
S3 access OK as arn:aws:iam::200283853008:user/<you>

[train] streaming s3://authenta-data-rnd/image-tampering-detection/processed-v1/train  ...
  torch Dataset: yes | samples: 1,827,437 (expected 1,827,437)
  sample 0: LAION-Mobile / authentic / LAION-Mobile/images/authentic/180493.jpeg (3264x2448 original)
  authentic sample 0: label=0, mask all zero ✓
  tampered  sample 1: label=1, mask has tampered pixels ✓
  batch 0: image=(8, 3, 512, 512), mask=(8, 1, 512, 512), edge_mask=(8, 1, 512, 512), label=(8,)  labels=[0, 1, 0, 1, ...]
  ...
[test] streaming s3://.../processed-v1/test  ...
  torch Dataset: yes | samples: 27,604 (expected 27,604)
  ...
PASS
```

**A3. Every loading path** — local copy and S3, directly and through the
training config, 0 and 4 workers, several roots, resume mid-epoch, both
dataset backends (~5 min)

```bash
python tests/smoke_test_all_paths.py
```

Expected ending:

```
PASS  1. local train, DataLoader shuffle=True, num_workers=0
PASS  2. local test, DataLoader shuffle=True, num_workers=4
PASS  3. local, list of two roots [train, test]
PASS  4. S3 train + test, direct, num_workers=4
PASS  5. training config, local dataroot (get_dataloaders_from_yml, StatefulDataLoader)
PASS  6. training config, s3:// dataroot + cache_dir
PASS  7. get_sample_from_yml (used by tests/overfit_single_batch.py)
PASS  8. build_forensics_datasets(data_format='mds')
PASS  9. module self-test: python -m authgenforge.data.forensics_mds_dataset
PASS  10. module self-test: python -m authgenforge.data.forensics_mds_dataloader
PASS  11. resume: StatefulDataLoader state_dict -> same next batch
PASS  12. folder backend (ForensicsDataset via build_dataloaders): label from full mask
12/12 passed
PASS
```

**A4. The converter itself** (builds a tiny fake dataset, ~20 s, no AWS needed)

```bash
python tests/smoke_test_mds.py
```

Ends with `PASS`.

**A5. Optional — full integrity check of the local copy** (reads every sample;
test ≈ 2 min, train ≈ 2.5 h)

```bash
python packages/mdsconverter/verify_mds_dataset.py --mds /home/ubuntu/data/processed/processed-v1/test \
    --decode-check 500 --spot-check 500 \
    --source-root /home/ubuntu/data/raw/test /home/ubuntu/data/extracted
```

Ends with `clean: labels agree with orig_path, masks present exactly for tampered samples, all checked samples decode/match.`
For train, replace `test` with `train` in all three paths.

---

## Option B — on a new machine (from scratch, streams from S3) · ~15 min

Needs Linux or macOS, `git`, the AWS CLI, and **Python 3.11** (3.12 also works;
3.13+ does not — `numpy==1.26.4` is pinned).

**B1. Get the code**

```bash
git clone -b processed-v1-dataset https://github.com/phospheneai/general-image-tampering-detecction.git
cd general-image-tampering-detecction
```

(Private repo: use your GitHub username and a personal access token when asked.)

**B2. Create the environment and install**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision
bash scripts/install_deps.sh
```

No `python3.11`? Use `uv` instead of the first line:
`pip install uv && uv venv --python 3.11 .venv`.

**B3. AWS access** — see [Before you start](#before-you-start-aws-access), then:

```bash
export AWS_PROFILE=<your-profile>
```

**B4. Run the checks**

```bash
python tests/smoke_test_s3.py      # real dataset from S3 -> PASS
python tests/smoke_test_mds.py     # converter on a fake dataset -> PASS
```

(`tests/smoke_test_all_paths.py` also needs the local copy; on a machine
without it run `python tests/smoke_test_all_paths.py --skip-s3 --local-root <path>`
after downloading one, or use Option A.)

---

## Load it yourself

Run from the repo root with the environment active.

**From S3** (shards download on demand into `cache_dir`):

```python
from torch.utils.data import DataLoader
from authgenforge.data.forensics_mds_dataset import ForensicsMDSDataset
from authgenforge.augmentations.presets import get_train_transforms

ds = ForensicsMDSDataset(
    "s3://authenta-data-rnd/image-tampering-detection/processed-v1/train",
    transform=get_train_transforms(crop_size=512),
    cache_dir="/home/ubuntu/data/mds_cache",
)
print(len(ds))                      # 1827437
print(ds.get_raw(0)["label_str"])   # authentic / tampered, plus dataset, orig_path, width, height, ...

loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)
batch = next(iter(loader))
print({k: tuple(v.shape) for k, v in batch.items()})
# {'image': (8, 3, 512, 512), 'mask': (8, 1, 512, 512), 'edge_mask': (8, 1, 512, 512), 'label': (8,)}
print(batch["label"])               # tensor([0, 1, 0, 1, ...])
```

**From the local copy** — same code, pass the folder and drop `cache_dir`:

```python
ds = ForensicsMDSDataset("/home/ubuntu/data/processed/processed-v1/train",
                         transform=get_train_transforms(crop_size=512))
```

- Use `get_val_transforms` (deterministic center crop) for the test split.
- `crop_size` can be anything (e.g. 384, 768) — no rebuild needed.
- Streaming with `shuffle=True` is correct but slow at first: each random sample
  may pull a different 512 MB shard. For a quick look use `shuffle=False`;
  for training, the cache fills once and later epochs read locally.
  Full train needs ~1.13 TB of free disk under `cache_dir`.

## Train on it

[`configs/normal/train_forensics_mds.yml`](configs/normal/train_forensics_mds.yml)
already points at the dataset (`data_format: mds`). To stream from S3 instead of
the local copy, change the two `dataroot` entries to
`s3://authenta-data-rnd/image-tampering-detection/processed-v1/train` and `.../test`
(`cache_dir` is set in the same file).

```bash
python -c "
from authgenforge.options.load import get_dataloaders_from_yml
tl, vl = get_dataloaders_from_yml('configs/normal/train_forensics_mds.yml')
print(len(tl.dataset), len(vl.dataset), {k: tuple(v.shape) for k, v in next(iter(tl)).items()})"
# 1827437 27604 {'image': (4, 3, 512, 512), 'mask': (4, 1, 512, 512), 'edge_mask': (4, 1, 512, 512), 'label': (4,)}
```

An actual training run additionally needs the DINOv3 weights
(`bash scripts/download_artifacts.sh`, Hugging Face access required) and
preferably a GPU; then `python tests/smoke_test_pipeline.py --config configs/normal/train_forensics_mds.yml`
runs a few real training steps.

## Where the code is

| file | what |
|---|---|
| [`authgenforge/data/forensics_mds_dataset.py`](authgenforge/data/forensics_mds_dataset.py) | `ForensicsMDSDataset` — the PyTorch `Dataset` (local or `s3://`) |
| [`authgenforge/data/forensics_mds_dataloader.py`](authgenforge/data/forensics_mds_dataloader.py) | `build_mds_dataloaders` — train/test `DataLoader`s used by training |
| [`authgenforge/options/load.py`](authgenforge/options/load.py) | `_DATASET_BACKENDS` — `data_format: mds` in a config selects it |
| [`configs/normal/train_forensics_mds.yml`](configs/normal/train_forensics_mds.yml) | training config for processed-v1 |
| [`tests/smoke_test_s3.py`](tests/smoke_test_s3.py), [`tests/smoke_test_all_paths.py`](tests/smoke_test_all_paths.py), [`tests/smoke_test_mds.py`](tests/smoke_test_mds.py) | the checks above |
| [`packages/mdsconverter/`](packages/mdsconverter/) | how processed-v1 was built |

## Troubleshooting

| you see | cause | fix |
|---|---|---|
| `cannot read s3://... (403) Forbidden` or `index.json not found!` | no S3 credentials, or the instance role is being used | `export AWS_PROFILE=<your-profile>` (see [AWS access](#before-you-start-aws-access)) |
| `ModuleNotFoundError: No module named 'authgenforge'` | not in the repo folder, or env not active | `cd` into the repo; `source ~/venv/bin/activate` (A) or `source .venv/bin/activate` (B) |
| `ModuleNotFoundError: No module named 'streaming'` / `'torchdata'` | dependencies not installed | `bash scripts/install_deps.sh` |
| numpy build errors during install | Python 3.13 or newer | use Python 3.11 (B2) |
| `UserWarning: 'set_vital' is deprecated` / `pin_memory ... no accelerator` | library notice / no GPU on this machine | harmless — ignore |
| first batch from S3 takes minutes | `shuffle=True` touching many shards | use `shuffle=False` for a quick check |
