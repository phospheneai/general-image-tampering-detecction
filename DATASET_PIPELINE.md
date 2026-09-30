# Dataset pipeline: validate → remediate → convert

Three standalone scripts in `packages/mdsconverter/`, each with its own
config in `configs/mds/`, chained into one DAG by `run_pipeline.py`. Any
stage can be toggled off.

```
validate_dataset.py ──▶ remediate_dataset.py ──▶ build_mds_dataset.py
 (decode every image     (move failed samples      (write MDS shards,
  + mask, CSV report,     to quarantine — dry       see MDS_DATASET.md)
  touch nothing)          run by default)
```

Two datasets need a one-off **stage 0 (prepare)** first, because their raw
form doesn't fit the `images/` + `masks/` layout. Both write a complete
dataset folder under `/home/ubuntu/data/extracted/` and never touch `raw/`:

| script | dataset | what it does |
|---|---|---|
| `prepare_compraise.py` | compRAISE | extracts `compRAISE_full.zip` (CRC-checked) → `images/authentic/` |
| `prepare_columbia.py` | columbia | copies images, converts `raw masks/tampered/<stem>_edgemask.jpg` → binary `masks/tampered/<stem>.png` (bright red = camera 1 near the splicing boundary = tampered) |

`_forensics_common.py` holds the discovery / mask-pairing / validation
logic all stages share, so they always agree on what a sample is.
[`configs/mds/datasets.yml`](configs/mds/datasets.yml) is the shared list of
datasets, their folders, their split, and the mask suffixes.

| Stage | Script | Config | What it does |
|---|---|---|---|
| — | — | `datasets.yml` | where each dataset lives, dataset → split, `mask_suffixes` |
| 1 validate | `validate_dataset.py` | `validate_dataset.yml` | decodes every image + mask, writes `/home/ubuntu/data/logs/validation_report.csv`. Read-only |
| 2 remediate | `remediate_dataset.py` | `remediate_dataset.yml` | moves failed samples (+ masks) to `/home/ubuntu/data/quarantine/<split>/<dataset>/...`. Dry run unless `--execute`; never overwrites |
| 3 convert | `build_mds_dataset.py` | `mds_dataset.yml` | writes `/home/ubuntu/data/processed/processed-v1/<split>/` |
| check | `verify_mds_dataset.py` | — | labels, masks, decoding, bytes vs originals |
| all | `run_pipeline.py` | `pipeline.yml` | 1 → 2 → 3, pausing for y/n if validate found failures |

## Failure reasons

| reason | meaning | usual fix |
|---|---|---|
| `unreadable`, `image-decode-invalid` | image can't be read or is truncated | remediate |
| `missing-mask` | no mask matched this tampered image | add the naming suffix to `mask_suffixes`; don't remediate |
| `ambiguous-mask` | more than one mask matched (`x.png` and `x.bmp`) | remove the duplicate mask |
| `mask-decode-invalid` | mask can't be decoded | remediate |
| `mask-size-mismatch` | mask and image sizes differ | check the release; remediate |
| `empty-mask` | tampered image whose mask marks nothing | remediate |

`remediate_dataset.yml` leaves `missing-mask` out of `reasons:` on purpose.

## Typical workflow

```bash
source ~/venv/bin/activate

# 0. one-off prepare (idempotent)
python packages/mdsconverter/prepare_compraise.py
python packages/mdsconverter/prepare_columbia.py

# 1. validate only, read the "failed by dataset" summary / the CSV
python packages/mdsconverter/run_pipeline.py --config configs/mds/pipeline.yml --only validate

# 2. review what remediate would move (dry run), then move for real
python packages/mdsconverter/remediate_dataset.py --config configs/mds/remediate_dataset.yml
python packages/mdsconverter/remediate_dataset.py --config configs/mds/remediate_dataset.yml --execute

# 3. smoke test: 500 samples per split to a scratch output
python packages/mdsconverter/build_mds_dataset.py --config configs/mds/mds_dataset.yml \
    --limit 500 --out /home/ubuntu/data/mds_smoke

# 4. convert
python packages/mdsconverter/run_pipeline.py --config configs/mds/pipeline.yml --only convert

# 5. verify each split (see MDS_DATASET.md), then upload
aws s3 sync /home/ubuntu/data/processed/processed-v1/ \
    s3://authenta-data-rnd/image-tampering-detection/processed-v1/ --only-show-errors
```

`run_pipeline.py --only validate` / `--skip remediate` override the
config's `enabled:` flags for one run. The interactive remediate gate
treats "no terminal attached" as "no".
