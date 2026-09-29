"""
MDS/streaming-shard backed sibling of forensics_dataset.py's
ForensicsDataset. Reads raw image + mask bytes out of MosaicML Streaming
(MDS) shards built by packages/mdsconverter/build_mds_dataset.py instead of
loose files on disk — same transforms and the same {image, mask, edge_mask}
return dict; only the byte source differs. Mirrors
ai-generated-video-detection's authgenvideo/data/video_mds_dataset.py.

Bytes in the shards are the original files, never re-encoded (JPEG
compression traces are part of the signal); decoding happens here, per
sample. Authentic samples are stored with empty mask bytes — an all-zero
mask the size of the image is built instead.

mds_root accepts one MDS split directory (e.g. <out>/train, written by
build_mds_dataset.py) or a list of them — multiple roots are combined as
separate streaming.Stream sources within one StreamingDataset and mixed
together, not physically merged on disk. Each root is either:
  - a local directory holding index.json + shard.*.mds, read in place, or
  - a remote URL (s3://bucket/prefix/<split>), streamed shard by shard into
    <cache_dir>/<bucket>/<prefix>/<split>. S3 credentials come from boto3's
    default chain (instance role, AWS_PROFILE, ...). Needs s3:GetObject.

Deliberately map-style (StreamingDataset.__getitem__ + __len__), not
StreamingDataset's own IterableDataset __iter__ path, same as the
reference: a plain torch DataLoader then gives duplicate-free multi-worker
loading (sampler-based index sharding) and works with StatefulDataLoader
and custom samplers. For a remote root, a random-order epoch touches every
shard, so the whole split ends up in cache_dir — leave cache_limit unset
(or at least as large as the split) for training, or the cache thrashes.

Linux (fork) only for num_workers > 0: StreamingDataset's shared memory is
registered by the process that builds the dataset, and spawned (Windows /
macOS) workers can't find it.

    python -m authgenforge.data.forensics_mds_dataset <mds_train_dir_or_s3_url> [...] [--cache-dir DIR]
"""

from __future__ import annotations

from urllib.parse import urlparse

from torch.utils.data import Dataset

from streaming import Stream, StreamingDataset

from authgenforge import *
from authgenforge.augmentations.presets import (
    get_train_transforms,
    get_val_transforms,
)
from authgenforge.data.forensics_dataset import _compute_edge_mask


# ==========================================
# Dataset
# ==========================================


class ForensicsMDSDataset(Dataset):
    """
    Forgery segmentation dataset over MDS shards — one augmented image +
    binary mask + boundary-band weight map per sample.

    See the module docstring for the mds_root / cache_dir conventions.
    """

    _MAX_DECODE_ATTEMPTS = 5

    def __init__(
        self,
        mds_root: str | list[str],
        transform,
        edge_kernel_size: int = 7,
        cache_dir: str | None = None,
        cache_limit: str | int | None = None,
    ):
        self.transform = transform
        self.edge_kernel_size = edge_kernel_size
        self.dataset = self._open(mds_root, cache_dir, cache_limit)

        roots = [mds_root] if isinstance(mds_root, str) else list(mds_root)
        streamed = any(_is_remote(r) for r in roots)

        print(
            f"[ForensicsMDSDataset] {len(self.dataset)} samples "
            f"across {len(roots)} MDS root(s)"
            f"{f'  |  cache={cache_dir}' if streamed else ''}  "
            # per-class counts aren't printed here — the label is a column
            # packed next to the image bytes, so counting would mean
            # reading every sample. verify_mds_dataset.py does that pass.
            f"(authentic/tampered breakdown: see verify_mds_dataset.py)"
        )

    @staticmethod
    def _open(mds_root, cache_dir, cache_limit) -> StreamingDataset:
        roots = [mds_root] if isinstance(mds_root, str) else list(mds_root)

        if not roots:
            raise ValueError(
                "mds_root must be a non-empty path or list of paths"
            )

        streams = []

        for root in roots:

            if _is_remote(root):

                if not cache_dir:
                    raise ValueError(
                        f"cache_dir is required to stream {root!r}"
                    )

                url = urlparse(root)

                streams.append(
                    Stream(
                        remote=root.rstrip("/"),
                        local=os.path.join(
                            cache_dir,
                            url.netloc,
                            url.path.strip("/"),
                        ),
                    )
                )

            else:
                streams.append(
                    Stream(local=root)
                )

        try:
            return StreamingDataset(
                streams=streams,
                shuffle=False,
                cache_limit=cache_limit,
            )
        except Exception as e:
            raise RuntimeError(
                f"failed to open MDS dataset at {roots!r}: {e}"
            ) from e

    def __len__(self) -> int:
        return len(self.dataset)

    def get_raw(self, index: int) -> dict:
        """
        The stored sample dict (bytes + metadata columns), undecoded.
        """

        return self.dataset[index]

    # A corrupt sample (bad bytes, shard I/O error) shouldn't kill a long
    # run: log its orig_path/dataset and fall back to a random other
    # sample, up to _MAX_DECODE_ATTEMPTS in a row. Same contract as the
    # reference's MDSVideoClassificationDataset.
    def __getitem__(self, index: int) -> dict:
        return self._getitem_with_fallback(
            index,
            self._MAX_DECODE_ATTEMPTS,
        )

    def _getitem_with_fallback(self, index: int, attempts_left: int) -> dict:
        try:
            return self._load_sample(index)
        except Exception as e:
            if attempts_left <= 1:
                raise RuntimeError(
                    f"Failed to decode {self._MAX_DECODE_ATTEMPTS} consecutive "
                    f"MDS samples (last index={index}) — dataset may have a "
                    f"systemic corruption issue, not just one bad sample"
                ) from e

            print(
                f"[ForensicsMDSDataset] WARNING: corrupted sample at "
                f"index={index} ({e}) — falling back to a different sample"
            )

            return self._getitem_with_fallback(
                random.randrange(len(self.dataset)),
                attempts_left - 1,
            )

    def _load_sample(self, index: int) -> dict:

        sample = self.dataset[index]  # shard-level I/O errors raise here too

        try:
            image = Image.open(
                io.BytesIO(sample["image"])
            ).convert("RGB")

            if len(sample["mask"]) > 0:
                mask = Image.open(
                    io.BytesIO(sample["mask"])
                ).convert("L")
            else:
                mask = Image.new("L", image.size, 0)
        except Exception as e:
            raise RuntimeError(
                f"corrupt image/mask bytes for "
                f"orig_path={sample['orig_path']!r} "
                f"(dataset={sample['dataset']!r})"
            ) from e

        image, mask = self.transform(image, mask)

        edge_mask = _compute_edge_mask(
            mask,
            self.edge_kernel_size,
        )

        return {
            "image": image,
            "mask": mask,
            "edge_mask": edge_mask,
        }


def _is_remote(root: str) -> bool:
    return "://" in root


def build_forensics_mds_datasets(
    train_dir: str | list[str],
    test_dir: str | list[str],
    crop_size: int = 512,
    cache_dir: str | None = None,
    cache_limit: str | int | None = None,
) -> tuple[ForensicsMDSDataset, ForensicsMDSDataset]:

    train_ds = ForensicsMDSDataset(
        train_dir,
        transform=get_train_transforms(crop_size=crop_size),
        cache_dir=cache_dir,
        cache_limit=cache_limit,
    )

    test_ds = ForensicsMDSDataset(
        test_dir,
        transform=get_val_transforms(crop_size=crop_size),
        cache_dir=cache_dir,
        cache_limit=cache_limit,
    )

    return train_ds, test_ds


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="MDS split dir(s) or s3:// URL(s)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--crop-size", type=int, default=512)
    args = ap.parse_args()

    ds = ForensicsMDSDataset(
        mds_root=args.roots,
        transform=get_val_transforms(crop_size=args.crop_size),
        cache_dir=args.cache_dir,
    )
    sample = ds[0]
    size = args.crop_size

    assert sample["image"].shape == (3, size, size), f"bad image shape: {sample['image'].shape}"
    assert sample["mask"].shape == (1, size, size), f"bad mask shape: {sample['mask'].shape}"
    assert sample["edge_mask"].shape == (1, size, size), f"bad edge_mask shape: {sample['edge_mask'].shape}"
    assert sample["mask"].min() >= 0.0 and sample["mask"].max() <= 1.0, "mask out of [0, 1]"

    for k, v in sample.items():
        print(f"{k:10s}: {tuple(v.shape)} {v.dtype}")
    print("PASS\n")
