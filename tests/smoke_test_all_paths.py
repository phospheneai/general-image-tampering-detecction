"""
Smoke test: every way the forensics datasets can be loaded, against the real
processed-v1 (local copy + S3) — both backends, direct and via the training
config, one or several roots, 0 or 4 DataLoader workers, the modules'
self-tests, and mid-epoch resume. Each case checks the batch contract:
{image (B,3,H,W) float32, mask (B,1,H,W), edge_mask (B,1,H,W), label (B,) int64 0/1},
with authentic (label 0) samples carrying an all-zero mask.

S3 cases need credentials with s3:GetObject on processed-v1 (e.g. AWS_PROFILE);
skip them with --skip-s3. Run from the project root:
    AWS_PROFILE=<profile> python tests/smoke_test_all_paths.py
    python tests/smoke_test_all_paths.py --skip-s3
    python tests/smoke_test_all_paths.py --local-root /path/to/processed-v1
"""
import argparse, os, shutil, subprocess, sys, tempfile, traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from authgenforge.augmentations.presets import get_train_transforms, get_val_transforms
from authgenforge.data.forensics_mds_dataset import ForensicsMDSDataset

ap = argparse.ArgumentParser()
ap.add_argument("--local-root", default="/home/ubuntu/data/processed/processed-v1")
ap.add_argument("--s3-root", default="s3://authenta-data-rnd/image-tampering-detection/processed-v1")
ap.add_argument("--skip-s3", action="store_true", help="skip the cases that stream from S3")
ARGS = ap.parse_args()

LOCAL, S3 = ARGS.local_root.rstrip("/"), ARGS.s3_root.rstrip("/")
KEYS = {"image", "mask", "edge_mask", "label"}
SCRATCH = None  # tempfile default dir
results = []


def check_batch(batch, crop=512):
    assert set(batch) == KEYS, sorted(batch)
    bs = batch["image"].shape[0]
    assert batch["image"].shape == (bs, 3, crop, crop) and batch["image"].dtype == torch.float32
    assert batch["mask"].shape == (bs, 1, crop, crop) and batch["edge_mask"].shape == (bs, 1, crop, crop)
    assert batch["label"].shape == (bs,) and batch["label"].dtype == torch.int64
    assert set(batch["label"].tolist()) <= {0, 1}
    assert (batch["mask"][batch["label"] == 0] == 0).all(), "authentic with non-zero mask"
    return batch["label"].tolist()


def case(name, s3=False):
    def deco(fn):
        if s3 and ARGS.skip_s3:
            results.append((name, "SKIP", "--skip-s3"))
            print(f"[SKIP] {name}", flush=True)
            return
        try:
            info = fn()
            results.append((name, "PASS", info or ""))
        except Exception as e:
            traceback.print_exc()
            results.append((name, "FAIL", f"{type(e).__name__}: {e}"[:160]))
        print(f"[{results[-1][1]}] {name}  {results[-1][2]}", flush=True)
    return deco


@case("1. local train, DataLoader shuffle=True, num_workers=0")
def _():
    ds = ForensicsMDSDataset(f"{LOCAL}/train", transform=get_train_transforms(crop_size=512))
    assert isinstance(ds, Dataset) and len(ds) == 1_827_437
    return check_batch(next(iter(DataLoader(ds, batch_size=8, shuffle=True, num_workers=0))))


@case("2. local test, DataLoader shuffle=True, num_workers=4")
def _():
    ds = ForensicsMDSDataset(f"{LOCAL}/test", transform=get_val_transforms(crop_size=512))
    assert len(ds) == 27_604
    return check_batch(next(iter(DataLoader(ds, batch_size=8, shuffle=True, num_workers=4))))


@case("3. local, list of two roots [train, test]")
def _():
    ds = ForensicsMDSDataset([f"{LOCAL}/train", f"{LOCAL}/test"], transform=get_val_transforms(crop_size=512))
    assert len(ds) == 1_827_437 + 27_604, len(ds)
    last = ds.get_raw(len(ds) - 1)
    assert last["split"] == "test", last["split"]
    check_batch(next(iter(DataLoader(ds, batch_size=8, shuffle=True, num_workers=4))))
    return f"len={len(ds):,}, last sample split={last['split']}"


@case("4. S3 train + test, direct, num_workers=4", s3=True)
def _():
    out = []
    with tempfile.TemporaryDirectory(dir=SCRATCH) as cache:
        for split, n in (("train", 1_827_437), ("test", 27_604)):
            ds = ForensicsMDSDataset(f"{S3}/{split}", transform=get_val_transforms(crop_size=512), cache_dir=cache)
            assert len(ds) == n
            out.append(check_batch(next(iter(DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)))))
    return out


@case("5. training config, local dataroot (get_dataloaders_from_yml, StatefulDataLoader)")
def _():
    from authgenforge.options.load import get_dataloaders_from_yml
    tl, vl = get_dataloaders_from_yml("configs/normal/train_forensics_mds.yml")
    assert type(tl).__name__ == "StatefulDataLoader", type(tl)
    a = check_batch(next(iter(tl)))
    b = check_batch(next(iter(vl)))
    return f"{type(tl).__name__} train={a} test={b}"


@case("6. training config, s3:// dataroot + cache_dir", s3=True)
def _():
    from authgenforge.options.load import get_dataloaders_from_yml
    from authgenforge.options.option_utils import parse_yml
    cfg = yaml.safe_load(open("configs/normal/train_forensics_mds.yml"))
    cache = tempfile.mkdtemp(dir=SCRATCH)
    cfg["cache_dir"] = cache
    cfg["datasets"]["train"]["dataroot"] = [f"{S3}/train"]
    cfg["datasets"]["test"]["dataroot"] = [f"{S3}/test"]
    p = os.path.join("configs/normal", "_check_s3.yml")  # same dir, so relative paths resolve as usual
    yaml.safe_dump(cfg, open(p, "w"))
    try:
        assert parse_yml(p)["datasets"]["train"]["dataroot"] == [f"{S3}/train"], parse_yml(p)["datasets"]["train"]["dataroot"]
        tl, vl = get_dataloaders_from_yml(p)
        assert len(tl.dataset) == 1_827_437 and len(vl.dataset) == 27_604
        # train loader shuffles (one shard per random sample) — read sample 0 directly;
        # the test loader reads in order, so a real batch only needs the first shard
        s = tl.dataset[0]
        assert set(s) == KEYS and s["label"].item() in (0, 1)
        return f"train sample0 label={s['label'].item()}, test batch={check_batch(next(iter(vl)))}"
    finally:
        os.remove(p)
        shutil.rmtree(cache, ignore_errors=True)


@case("7. get_sample_from_yml (used by tests/overfit_single_batch.py)")
def _():
    from authgenforge.options.load import get_sample_from_yml
    return check_batch(get_sample_from_yml("configs/normal/train_forensics_mds.yml"))


@case("8. build_forensics_datasets(data_format='mds')")
def _():
    from authgenforge.data.forensics_dataset import build_forensics_datasets
    tr, te = build_forensics_datasets(f"{LOCAL}/train", f"{LOCAL}/test", crop_size=384, data_format="mds")
    check_batch(next(iter(DataLoader(te, batch_size=4, num_workers=0))), crop=384)
    return f"train={len(tr):,} test={len(te):,} (crop 384)"


@case("9. module self-test: python -m authgenforge.data.forensics_mds_dataset")
def _():
    r = subprocess.run([sys.executable, "-m", "authgenforge.data.forensics_mds_dataset", f"{LOCAL}/test"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "PASS" in r.stdout, r.stdout[-500:] + r.stderr[-800:]
    return [l for l in r.stdout.splitlines() if l.startswith("label")][0]


@case("10. module self-test: python -m authgenforge.data.forensics_mds_dataloader")
def _():
    r = subprocess.run([sys.executable, "-m", "authgenforge.data.forensics_mds_dataloader"],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "[train]" in r.stdout and "[test]" in r.stdout, r.stdout[-500:] + r.stderr[-800:]
    return " | ".join(l.strip() for l in r.stdout.splitlines() if l.startswith("["))[-200:]


@case("11. resume: StatefulDataLoader state_dict -> same next batch")
def _():
    from torchdata.stateful_dataloader import StatefulDataLoader
    ds = ForensicsMDSDataset(f"{LOCAL}/test", transform=get_val_transforms(crop_size=128))
    torch.manual_seed(0)
    a = StatefulDataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    it = iter(a); next(it); next(it)
    state = a.state_dict()
    expected = next(it)
    b = StatefulDataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    b.load_state_dict(state)
    got = next(iter(b))
    assert torch.equal(expected["image"], got["image"]) and torch.equal(expected["label"], got["label"])
    return f"resumed batch identical, labels={got['label'].tolist()}"


@case("12. folder backend (ForensicsDataset via build_dataloaders): label from full mask")
def _():
    from authgenforge.data.dataloader import build_dataloaders
    with tempfile.TemporaryDirectory(dir=SCRATCH) as root:
        os.makedirs(f"{root}/images"); os.makedirs(f"{root}/masks")
        rng = np.random.default_rng(0)
        for i in range(8):
            Image.fromarray(rng.integers(0, 255, (600, 600, 3), dtype=np.uint8)).save(f"{root}/images/x{i}.png")
            m = np.zeros((600, 600), np.uint8)
            if i % 2:
                m[10:40, 10:40] = 255  # small corner region: a 512 crop may miss it, label must not
            Image.fromarray(m).save(f"{root}/masks/x{i}.png")
        tl, vl = build_dataloaders(root, root, crop_size=512, batch_size=8, num_workers=0, stateful=False,
                                   pin_memory=False, data_format="folder")
        b = next(iter(vl))
        assert set(b) == KEYS, sorted(b)
        assert sorted(b["label"].tolist()) == [0, 0, 0, 0, 1, 1, 1, 1], b["label"]
        return f"labels={b['label'].tolist()} (4 authentic / 4 tampered)"


print("\n" + "=" * 90)
for name, status, info in results:
    print(f"{status:4s}  {name}")
print("=" * 90)
fails = [r for r in results if r[1] == "FAIL"]
skipped = sum(r[1] == "SKIP" for r in results)
print(f"{len(results) - len(fails) - skipped}/{len(results)} passed" + (f", {skipped} skipped" if skipped else ""))
print("PASS" if not fails else "FAIL")
sys.exit(1 if fails else 0)
