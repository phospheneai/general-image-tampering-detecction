"""
Shared helpers for the forensics dataset pipeline scripts — validate_dataset.py,
remediate_dataset.py, build_mds_dataset.py, verify_mds_dataset.py,
run_pipeline.py. Factored out once so every stage agrees on what "dataset",
"split", "label" and "mask" mean for a given file — if discovery or mask
pairing lived separately in each script, validate_dataset.py could flag a
file that build_mds_dataset.py doesn't even consider "the same sample".

Expected on-disk layout, one folder per source dataset under data_root:

    <data_root>/<dataset>/images/authentic/<name>.<ext>
    <data_root>/<dataset>/images/tampered/<name>.<ext>
    <data_root>/<dataset>/mask/tampered/<name>[<suffix>].<ext>   (or masks/)

Authentic images carry no mask on disk (an all-zero mask is implied and
built at load time); every tampered image must pair with exactly one mask
in mask/tampered/ by filename stem, optionally followed by one of the
configured mask_suffixes (e.g. "_gt", "_mask").

Not a CLI entry point itself — imported by the scripts above.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import yaml

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a project dep, but don't hard-fail
    def tqdm(it=None, **kw):
        return it if it is not None else _NullBar()

    class _NullBar:
        def update(self, *a, **kw): pass
        def close(self): pass

LABEL_TO_INT = {"authentic": 0, "tampered": 1}

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"]
MASK_EXTS = [".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".gif"]

# Tried in order when pairing a tampered image with its mask: "" first (exact
# stem match), then the common suffixes different dataset releases use.
DEFAULT_MASK_SUFFIXES = ["", "_gt", "_mask", "_GT", "_Mask", "-mask"]

# Folder names looked for under each dataset root. First one that exists wins.
IMAGES_DIRNAME = "images"
MASK_DIRNAMES = ("mask", "masks")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def load_yaml_config(path: str, path_keys: tuple[str, ...] = ()) -> dict:
    """Loads yaml, resolving any of `path_keys` relative to the yaml file's
    own directory — so a config works the same regardless of the cwd it's
    launched from. `datasets:` entries with an explicit `path:` are resolved
    the same way.

    `datasets_config: <file>` pulls data_root/datasets/mask_suffixes/
    image_exts/mask_exts from a shared yaml (configs/mds/datasets.yml), so
    validate and convert can never disagree about which datasets exist or
    which split they belong to. Keys set in this file itself win."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = os.path.dirname(os.path.abspath(path))

    def _resolve(p: str) -> str:
        p = os.path.expanduser(p)
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

    if cfg.get("datasets_config"):
        shared = load_yaml_config(_resolve(cfg.pop("datasets_config")), path_keys=("data_root",))
        for key in ("data_root", "datasets", "mask_suffixes", "image_exts", "mask_exts"):
            if key in shared and cfg.get(key) is None:
                cfg[key] = shared[key]

    for key in path_keys:
        if cfg.get(key):
            cfg[key] = _resolve(cfg[key])
    for name, spec in (cfg.get("datasets") or {}).items():
        if isinstance(spec, dict) and spec.get("path"):
            spec["path"] = _resolve(spec["path"])
    return cfg


def cfg_picker(cli_ns, cfg: dict):
    """Returns pick(cli_val, cfg_key, default) -> cli value if given (not
    None), else cfg[cfg_key] if present, else default. The standard "CLI
    flag overrides this one config key" merge used by every stage."""
    def pick(cli_val, cfg_key, default):
        if cli_val is not None:
            return cli_val
        return cfg.get(cfg_key, default)
    return pick


@dataclass(slots=True)
class DatasetSpec:
    name: str
    split: str
    path: str  # absolute path to <data_root>/<name> (or the explicit override)


def parse_datasets(data_root: str | None, datasets_cfg: dict) -> list[DatasetSpec]:
    """`datasets:` map -> [DatasetSpec, ...]. Each value is either a bare
    split name (folder assumed at <data_root>/<name>) or a dict
    {split: ..., path: ...} for a dataset that lives somewhere else. A split
    of null/"skip" drops the dataset without having to delete its entry."""
    specs = []
    for name, spec in (datasets_cfg or {}).items():
        if isinstance(spec, dict):
            split, path = spec.get("split"), spec.get("path")
        else:
            split, path = spec, None
        if split in (None, "", "skip"):
            continue
        if not path:
            if not data_root:
                raise ValueError(f"dataset {name!r} has no path: and no data_root is set")
            path = os.path.join(data_root, name)
        specs.append(DatasetSpec(name=name, split=str(split), path=os.path.abspath(os.path.expanduser(path))))
    return specs


def parse_datasets_arg(data_root: str | None, items: list[str]) -> list[DatasetSpec]:
    """CLI form: 'NAME:split' or bare 'NAME' (split=train). Names never
    contain a colon, so splitting on the first one is unambiguous even on
    Windows (the path comes from data_root, not from this argument)."""
    as_dict = {}
    for item in items:
        name, _, split = item.partition(":")
        as_dict[name] = split or "train"
    return parse_datasets(data_root, as_dict)


def filter_split(specs: list[DatasetSpec], split: str | None) -> list[DatasetSpec]:
    if not split:
        return specs
    kept = [s for s in specs if s.split == split]
    if not kept:
        raise ValueError(f"split {split!r} not found (have: {sorted({s.split for s in specs})})")
    return kept


def normalize_exts(raw, default: list[str]) -> set[str]:
    raw = raw if raw else default
    if isinstance(raw, str):
        raw = raw.split(",")
    return {e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
            for e in raw if e.strip()}


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

@dataclass(slots=True)
class Sample:
    """One discovered image + its folder-derived metadata. `slots=True`
    keeps per-sample memory small — tampCOCO + LAION alone are ~1.6M
    samples held in memory at once during the shuffle.

    mask_path is "" for authentic images, and also "" for a tampered image
    whose mask couldn't be found — mask_error then says why, so the
    validate stage can report it instead of discovery silently dropping it."""
    path: str
    rel_path: str        # relative to data_root-level: "<dataset>/images/<label>/<file>"
    mask_path: str
    mask_rel_path: str
    split: str
    dataset: str
    label_str: str
    mask_error: str = ""


def _list_files(d: str, exts: set[str]) -> list[str]:
    """Recursive — some releases nest images in per-category subfolders
    under authentic/ or tampered/."""
    out = []
    for dirpath, _dirnames, filenames in os.walk(d):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def _find_subdir(parent: str, names) -> str | None:
    """Case-insensitive lookup of the first of `names` that exists under parent."""
    if not os.path.isdir(parent):
        return None
    entries = {e.lower(): e for e in os.listdir(parent)}
    for n in ([names] if isinstance(names, str) else names):
        hit = entries.get(n.lower())
        if hit and os.path.isdir(os.path.join(parent, hit)):
            return os.path.join(parent, hit)
    return None


def _build_mask_index(mask_dir: str, mask_exts: set[str]) -> dict[str, list[str]]:
    """{stem: [path, ...]} over every mask file under mask_dir. A list, not
    a single path, so two masks sharing a stem (x.png + x.bmp) surface as an
    'ambiguous mask' error rather than one silently winning."""
    index: dict[str, list[str]] = {}
    for p in _list_files(mask_dir, mask_exts):
        index.setdefault(os.path.splitext(os.path.basename(p))[0], []).append(p)
    return index


def _pair_mask(stem: str, index: dict[str, list[str]], suffixes: list[str]) -> tuple[str, str]:
    """(mask_path, error). Tries stem+suffix for each suffix in order; the
    first suffix with any hit decides."""
    for suf in suffixes:
        hits = index.get(stem + suf)
        if not hits:
            continue
        if len(hits) > 1:
            return "", f"ambiguous-mask: {len(hits)} candidates for stem {stem + suf!r}"
        return hits[0], ""
    return "", "missing-mask"


@dataclass(slots=True)
class DiscoveryStats:
    orphan_masks: int = 0  # masks no tampered image claimed — usually a naming mismatch worth a look


def discover(specs: list[DatasetSpec], image_exts: set[str], mask_exts: set[str],
             mask_suffixes: list[str]) -> tuple[list[Sample], dict[str, DiscoveryStats]]:
    samples: list[Sample] = []
    stats: dict[str, DiscoveryStats] = {}
    for spec in specs:
        st = stats.setdefault(spec.name, DiscoveryStats())
        if not os.path.isdir(spec.path):
            print(f"WARNING: dataset folder not found, skipping: {spec.name} -> {spec.path}", file=sys.stderr)
            continue

        images_dir = _find_subdir(spec.path, IMAGES_DIRNAME)
        if images_dir is None:
            print(f"WARNING: no images/ folder under {spec.path}, skipping {spec.name}", file=sys.stderr)
            continue
        auth_dir = _find_subdir(images_dir, "authentic")
        tamp_dir = _find_subdir(images_dir, "tampered")
        mask_root = _find_subdir(spec.path, MASK_DIRNAMES)
        mask_dir = _find_subdir(mask_root, "tampered") if mask_root else None

        auth_files = _list_files(auth_dir, image_exts) if auth_dir else []
        tamp_files = _list_files(tamp_dir, image_exts) if tamp_dir else []
        mask_index = _build_mask_index(mask_dir, mask_exts) if mask_dir else {}
        if tamp_files and mask_dir is None:
            print(f"WARNING: {spec.name} has {len(tamp_files)} tampered images but no "
                  f"mask/tampered/ folder — every one will be reported as missing-mask", file=sys.stderr)

        # rel paths are relative to the dataset folder's *parent*, so they
        # always start with the dataset name — unambiguous across datasets
        # and matches <data_root>/<rel_path> for the normal layout.
        rel_base = os.path.dirname(spec.path)

        for p in auth_files:
            samples.append(Sample(
                path=p, rel_path=os.path.relpath(p, rel_base), mask_path="", mask_rel_path="",
                split=spec.split, dataset=spec.name, label_str="authentic",
            ))

        claimed: set[str] = set()
        n_missing = 0
        for p in tamp_files:
            stem = os.path.splitext(os.path.basename(p))[0]
            mpath, err = _pair_mask(stem, mask_index, mask_suffixes)
            if mpath:
                claimed.add(mpath)
            else:
                n_missing += 1
            samples.append(Sample(
                path=p, rel_path=os.path.relpath(p, rel_base),
                mask_path=mpath, mask_rel_path=os.path.relpath(mpath, rel_base) if mpath else "",
                split=spec.split, dataset=spec.name, label_str="tampered", mask_error=err,
            ))

        st.orphan_masks = sum(len(v) for v in mask_index.values()) - len(claimed)
        print(f"  {spec.name} [{spec.split}]: {len(auth_files)} authentic, {len(tamp_files)} tampered"
              f"{f', {n_missing} without a usable mask' if n_missing else ''}"
              f"{f', {st.orphan_masks} orphan masks' if st.orphan_masks else ''}")
    return samples, stats


# --------------------------------------------------------------------------
# Validation (PIL) — shared by validate_dataset.py and build_mds_dataset.py's
# own optional inline --validate pass
# --------------------------------------------------------------------------

def _open_fully(path: str):
    """Decodes every pixel, not just the header — Image.open() alone is
    lazy and happily 'opens' a truncated JPEG. Returns the loaded image."""
    from PIL import Image
    im = Image.open(path)
    im.load()
    return im


def validate_sample(sample: Sample, check_mask_content: bool = True) -> tuple[bool, str, int, int, int]:
    """(ok, reason, filesize_bytes, width, height). Covers every failure
    mode the stages need to agree on: unreadable/undecodable image, and for
    tampered samples a missing/ambiguous/undecodable mask, a mask whose size
    doesn't match its image, or an all-zero mask (tampered but nothing
    marked — would train as a false 'authentic')."""
    from PIL import Image
    import numpy as np

    try:
        filesize = os.path.getsize(sample.path)
    except OSError as e:
        return False, f"unreadable: {e}", 0, -1, -1
    try:
        im = _open_fully(sample.path)
        width, height = im.size
    except Exception as e:
        return False, f"image-decode-invalid: {str(e)[:200]}", filesize, -1, -1

    if sample.label_str != "tampered":
        return True, "", filesize, width, height

    if sample.mask_error:
        return False, sample.mask_error, filesize, width, height
    try:
        mask = _open_fully(sample.mask_path)
    except Exception as e:
        return False, f"mask-decode-invalid: {str(e)[:200]}", filesize, width, height
    if mask.size != (width, height):
        return False, f"mask-size-mismatch: image {width}x{height} vs mask {mask.size[0]}x{mask.size[1]}", \
            filesize, width, height
    if check_mask_content:
        # same threshold MaskToTensor applies at train time (to_tensor > 0.5)
        arr = np.asarray(mask.convert("L"))
        if not (arr > 127).any():
            return False, "empty-mask: tampered image with no forged pixels marked", filesize, width, height
    return True, "", filesize, width, height


def image_size(path: str) -> tuple[int, int]:
    """Header-only (cheap) width/height, for the no-validate build path."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return -1, -1


# --------------------------------------------------------------------------
# CLI plumbing shared by validate_dataset.py and build_mds_dataset.py
# --------------------------------------------------------------------------

def add_dataset_args(ap) -> None:
    ap.add_argument("--config", default=None, help="Path to a yaml config. Any other flag passed "
                                                     "alongside it overrides that one key.")
    ap.add_argument("--split", default=None, help="Only process datasets assigned to this split "
                                                    "(e.g. --split train).")
    ap.add_argument("--data-root", default=None,
                     help="Folder holding one subfolder per dataset (<data-root>/<dataset>/images/...).")
    ap.add_argument("--datasets", nargs="+", default=None,
                     help="One or more 'NAME:split' (or bare 'NAME' -> train) folder names under "
                          "--data-root. Overrides --config's datasets: map if both are given.")
    ap.add_argument("--only-datasets", nargs="+", default=None,
                     help="Restrict to these dataset names (after --split filtering) — handy for "
                          "re-running one dataset without editing the config.")
    ap.add_argument("--image-exts", default=None, help=f"Comma-separated. Default: {','.join(IMAGE_EXTS)}")
    ap.add_argument("--mask-exts", default=None, help=f"Comma-separated. Default: {','.join(MASK_EXTS)}")
    ap.add_argument("--mask-suffixes", nargs="*", default=None,
                     help=f"Stem suffixes tried in order when pairing a mask. Default: {DEFAULT_MASK_SUFFIXES}")


def resolve_dataset_args(ap, cli, cfg: dict):
    """-> (specs, image_exts, mask_exts, mask_suffixes). Calls ap.error()
    on bad input, same as the rest of argparse."""
    pick = cfg_picker(cli, cfg)
    data_root = pick(cli.data_root, "data_root", None)
    if data_root:
        data_root = os.path.abspath(os.path.expanduser(data_root))
    try:
        if cli.datasets:
            specs = parse_datasets_arg(data_root, cli.datasets)
        elif cfg.get("datasets"):
            specs = parse_datasets(data_root, cfg["datasets"])
        else:
            ap.error("no datasets given (either --datasets or `datasets:` in --config)")
        specs = filter_split(specs, cli.split)
    except ValueError as e:
        ap.error(str(e))
    if cli.only_datasets:
        wanted = set(cli.only_datasets)
        unknown = wanted - {s.name for s in specs}
        if unknown:
            ap.error(f"--only-datasets: not in the (split-filtered) dataset list: {sorted(unknown)}")
        specs = [s for s in specs if s.name in wanted]
    image_exts = normalize_exts(pick(cli.image_exts, "image_exts", None), IMAGE_EXTS)
    mask_exts = normalize_exts(pick(cli.mask_exts, "mask_exts", None), MASK_EXTS)
    mask_suffixes = pick(cli.mask_suffixes, "mask_suffixes", None) or DEFAULT_MASK_SUFFIXES
    return specs, image_exts, mask_exts, list(mask_suffixes)
