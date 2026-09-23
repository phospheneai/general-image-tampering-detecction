#!/usr/bin/env python3
"""
Stage 3 of the forensics dataset pipeline — combines every configured
dataset folder into one MosaicML Streaming (MDS) dataset per split, ready to
feed authgenforge/data/forensics_mds_dataset.py at train time. Usually run
after validate_dataset.py (+ optionally remediate_dataset.py) — see
run_pipeline.py to run all three together. This script also has its own
--validate pass (on by default) as a standalone safety net, so it's still
correct to run on its own.

Input layout (see _forensics_common.py):
    <data_root>/<dataset>/images/authentic/<file>
    <data_root>/<dataset>/images/tampered/<file>
    <data_root>/<dataset>/mask/tampered/<file>

Which split a dataset lands in comes from the config's `datasets:` map
(DATASETS.md's Train/Test tables), not from the folder layout — so the same
raw tree can be re-split without moving files.

Bytes are stored exactly as they are on disk — never decoded and
re-encoded. For forgery detection this matters: re-saving a JPEG would
overwrite the very compression traces (double-JPEG, quantization tables)
the model is supposed to learn from. The mask column is empty bytes for
authentic images; the reader builds an all-zero mask for those.

Balance + shuffle: within each split, authentic and tampered samples are
each shuffled independently (seeded) and then interleaved proportionally
(Bresenham-style), so every physical MDS shard holds an authentic:tampered
ratio matching the split's global ratio. Each class's shuffle also mixes
datasets, so no shard is (say) all tampCOCO.

Speed: reading bytes (+ optional PIL validation) runs in a worker pool;
results come back in write order (Pool.imap, not imap_unordered) so shard
balance is exact, while MDSWriter consumes them in the main process.

Usage:
    # config-file driven — processes every split in the yaml's datasets: map
    python packages/mdsconverter/build_mds_dataset.py --config configs/mds/mds_dataset.yml

    # one split per invocation
    python packages/mdsconverter/build_mds_dataset.py --config configs/mds/mds_dataset.yml --split train
    python packages/mdsconverter/build_mds_dataset.py --config configs/mds/mds_dataset.yml --split test

    # plain CLI, no yaml
    python packages/mdsconverter/build_mds_dataset.py --data-root D:/forensics_raw \\
        --datasets CASIAv2:train Columbia:test --out D:/forensics_mds

    # quick smoke test: 500 samples per split
    python packages/mdsconverter/build_mds_dataset.py --config configs/mds/mds_dataset.yml --limit 500

Requires: pip install mosaicml-streaming pillow numpy tqdm pyyaml
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import dataclass

from streaming import MDSWriter

from _forensics_common import (
    LABEL_TO_INT, Sample, add_dataset_args, cfg_picker, discover, image_size,
    load_yaml_config, resolve_dataset_args, tqdm, validate_sample,
)

COLUMNS = {
    "image": "bytes",
    "mask": "bytes",            # b"" for authentic images
    "label": "int",             # 0 authentic / 1 tampered (LABEL_TO_INT)
    "label_str": "str",
    "dataset": "str",
    "split": "str",
    "orig_path": "str",         # "<dataset>/images/<label>/<file>", relative to data_root
    "mask_orig_path": "str",    # "" for authentic images
    "ext": "str",
    "mask_ext": "str",
    "width": "int",
    "height": "int",
    "filesize_bytes": "int",
}


# --------------------------------------------------------------------------
# Balance + shuffle
# --------------------------------------------------------------------------

def fair_interleave(a: list, b: list) -> list:
    """Merge two lists preserving each one's internal order, distributing
    the shorter list as evenly as possible across the longer one so every
    prefix (and therefore every shard) stays close to the global a:b ratio."""
    la, lb = len(a), len(b)
    total = la + lb
    out = []
    ia = ib = 0
    for i in range(total):
        target_a = round((i + 1) * la / total) if total else 0
        if ia < target_a and ia < la:
            out.append(a[ia]); ia += 1
        else:
            out.append(b[ib]); ib += 1
    return out


def balance_and_shuffle(samples: list[Sample], seed: int) -> list[Sample]:
    authentic = [s for s in samples if s.label_str == "authentic"]
    tampered = [s for s in samples if s.label_str == "tampered"]
    random.Random(seed).shuffle(authentic)
    random.Random(seed + 1).shuffle(tampered)
    return fair_interleave(authentic, tampered)


# --------------------------------------------------------------------------
# Per-sample processing (runs in worker pool)
# --------------------------------------------------------------------------

_VALIDATE_ENABLED = True
_CHECK_MASK_CONTENT = True


def _init_worker(validate_enabled: bool, check_mask_content: bool):
    global _VALIDATE_ENABLED, _CHECK_MASK_CONTENT
    _VALIDATE_ENABLED = validate_enabled
    _CHECK_MASK_CONTENT = check_mask_content


@dataclass(slots=True)
class ProcessResult:
    """Exactly one of `row`/`reason` is meaningful, decided by `ok`.
    write_split() is the single place failures get logged and counted."""
    ok: bool
    row: dict | None = None
    path: str = ""
    reason: str = ""


def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def process_one(sample: Sample) -> ProcessResult:
    path = sample.path
    if sample.label_str == "tampered" and not sample.mask_path:
        # never write a tampered sample without its mask, even with
        # --no-validate — it would silently train as all-zero ground truth
        return ProcessResult(ok=False, path=path, reason=sample.mask_error or "missing-mask")

    if _VALIDATE_ENABLED:
        ok, reason, _size, width, height = validate_sample(sample, _CHECK_MASK_CONTENT)
        if not ok:
            return ProcessResult(ok=False, path=path, reason=reason)
    else:
        width, height = image_size(path)

    try:
        image_bytes = _read(path)
        mask_bytes = _read(sample.mask_path) if sample.mask_path else b""
    except OSError as e:
        return ProcessResult(ok=False, path=path, reason=f"unreadable: {e}")

    row = {
        "image": image_bytes,
        "mask": mask_bytes,
        "label": LABEL_TO_INT[sample.label_str],
        "label_str": sample.label_str,
        "dataset": sample.dataset,
        "split": sample.split,
        # forward slashes so paths recorded on Windows still parse on Linux
        "orig_path": sample.rel_path.replace(os.sep, "/"),
        "mask_orig_path": sample.mask_rel_path.replace(os.sep, "/"),
        "ext": os.path.splitext(path)[1].lower(),
        "mask_ext": os.path.splitext(sample.mask_path)[1].lower() if sample.mask_path else "",
        "width": width,
        "height": height,
        "filesize_bytes": len(image_bytes),
    }
    return ProcessResult(ok=True, row=row)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

@contextlib.contextmanager
def _writer_out_path(out_dir: str):
    """Yields a path MDSWriter accepts for out_dir. MDSWriter urlparse()s
    its `out` to pick an uploader, so a Windows path like D:\\mds\\train
    parses as URL scheme "d" and is rejected as an unknown cloud provider.
    A relative path has no scheme — so on Windows, chdir into out_dir's
    parent for the duration of the write and hand over just the basename."""
    if os.name != "nt":
        yield out_dir
        return
    prev = os.getcwd()
    os.chdir(os.path.dirname(out_dir))
    try:
        yield os.path.basename(out_dir)
    finally:
        os.chdir(prev)


def write_split(split: str, order: list[Sample], out_dir: str, args) -> tuple[int, int]:
    written = skipped = 0
    label_counts = {"authentic": 0, "tampered": 0}
    per_dataset: dict[str, dict[str, int]] = {}
    t0 = time.time()

    failed_path = f"{os.path.normpath(out_dir)}_failed.csv"
    with open(failed_path, "w", newline="", encoding="utf-8") as failed_f:
        failed_writer = csv.writer(failed_f)
        failed_writer.writerow(["path", "reason"])

        with _writer_out_path(out_dir) as writer_out, \
                MDSWriter(out=writer_out, columns=COLUMNS, compression=args.compression,
                          hashes=args.hashes or None, size_limit=args.shard_size_mb * (1 << 20)) as writer:
            init_args = (args.validate, args.check_mask_content)
            with mp.Pool(args.num_workers, initializer=_init_worker, initargs=init_args) as pool:
                for result in tqdm(pool.imap(process_one, order, chunksize=args.chunksize),
                                    total=len(order), desc=f"[{split}] writing", unit="img"):
                    if not result.ok:
                        skipped += 1
                        failed_writer.writerow([result.path, result.reason])
                        continue
                    writer.write(result.row)
                    written += 1
                    label = result.row["label_str"]
                    label_counts[label] += 1
                    per_dataset.setdefault(result.row["dataset"], {"authentic": 0, "tampered": 0})[label] += 1
                    del result

    dt = time.time() - t0
    print(f"[{split}] wrote {written} samples ({label_counts['authentic']} authentic / "
          f"{label_counts['tampered']} tampered), skipped {skipped}, in {dt:.1f}s "
          f"({written / dt if dt > 0 else 0:.1f} samples/s) -> {out_dir}")
    for name, c in sorted(per_dataset.items()):
        print(f"    {name}: {c['authentic']} authentic / {c['tampered']} tampered")
    if skipped:
        print(f"[{split}] {skipped} failures logged to {failed_path}")
    else:
        os.remove(failed_path)  # nothing to report -- don't leave a header-only file around
    return written, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_args(ap)
    ap.add_argument("--out", default=None, help="Output root; one MDS dataset per split at <out>/<split>/")
    ap.add_argument("--shard-size-mb", type=int, default=None, help="Target shard size in MB. Default: 256")
    ap.add_argument("--compression", default=None,
                     help="e.g. 'zstd:7'. Off by default — JPEG/PNG are already compressed.")
    ap.add_argument("--hashes", nargs="*", default=None, help="e.g. --hashes sha1 xxh64. Off by default.")
    ap.add_argument("--num-workers", type=int, default=None, help="Producer pool size. Default: os.cpu_count()")
    ap.add_argument("--chunksize", type=int, default=None,
                     help="Pool.imap chunksize. Default: 16 (images are small; raise/lower with RAM).")
    ap.add_argument("--seed", type=int, default=None, help="Default: 42")
    ap.add_argument("--validate", dest="validate", action="store_true", default=None,
                     help="Fully decode every image + mask with PIL before writing; failures are skipped and "
                          "logged to <out>/<split>_failed.csv. Default: on.")
    ap.add_argument("--no-validate", dest="validate", action="store_false", default=None,
                     help="Skip decode validation (safe if validate_dataset.py + remediate already ran). "
                          "Tampered samples with no mask are still always skipped.")
    ap.add_argument("--no-check-mask-content", dest="check_mask_content", action="store_false", default=None,
                     help="With --validate: don't reject tampered samples whose mask is all zeros.")
    ap.add_argument("--limit", type=int, default=None, help="Debug: cap total samples per split before writing.")
    cli = ap.parse_args()

    cfg = load_yaml_config(cli.config, path_keys=("out", "data_root")) if cli.config else {}
    pick = cfg_picker(cli, cfg)

    out = pick(cli.out, "out", None)
    if not out:
        ap.error("--out is required (either as a flag or `out:` in --config)")
    specs, image_exts, mask_exts, mask_suffixes = resolve_dataset_args(ap, cli, cfg)

    class Args:
        pass
    args = Args()
    args.out = os.path.abspath(os.path.expanduser(out))
    args.shard_size_mb = pick(cli.shard_size_mb, "shard_size_mb", 256)
    args.compression = pick(cli.compression, "compression", None)
    args.hashes = pick(cli.hashes, "hashes", None)
    args.num_workers = pick(cli.num_workers, "num_workers", os.cpu_count())
    args.chunksize = pick(cli.chunksize, "chunksize", 16)
    args.seed = pick(cli.seed, "seed", 42)
    args.validate = pick(cli.validate, "validate", True)
    args.check_mask_content = pick(cli.check_mask_content, "check_mask_content", True)
    args.limit = pick(cli.limit, "limit", None)

    print(f"discovering {len(specs)} dataset(s), mask suffixes={mask_suffixes} ...")
    samples, _stats = discover(specs, image_exts, mask_exts, mask_suffixes)
    print(f"discovered {len(samples)} total images")
    if not samples:
        print("nothing found — check data_root / datasets: / folder layout", file=sys.stderr)
        sys.exit(1)

    by_split: dict[str, list[Sample]] = {}
    for s in samples:
        by_split.setdefault(s.split, []).append(s)

    os.makedirs(args.out, exist_ok=True)
    grand_written = grand_skipped = 0
    for split, split_samples in sorted(by_split.items()):
        out_dir = os.path.join(args.out, split)
        if os.path.exists(os.path.join(out_dir, "index.json")):
            # MDSWriter refuses a non-empty dir anyway; fail with a clearer message
            print(f"ERROR: {out_dir} already holds an MDS dataset — delete it or pick another --out",
                  file=sys.stderr)
            sys.exit(1)
        n_auth = sum(1 for s in split_samples if s.label_str == "authentic")
        print(f"\n=== split '{split}': {len(split_samples)} images "
              f"({n_auth} authentic / {len(split_samples) - n_auth} tampered) ===")
        order = balance_and_shuffle(split_samples, args.seed)
        if args.limit:
            order = order[: args.limit]
        written, skipped = write_split(split, order, out_dir, args)
        grand_written += written
        grand_skipped += skipped

    print(f"\ndone. {grand_written} samples written, {grand_skipped} skipped. output: {args.out}")


if __name__ == "__main__":
    main()
