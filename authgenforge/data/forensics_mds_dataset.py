from __future__ import annotations

import bisect
import io
import json
import os

from torch.utils.data import Dataset

from authgenforge import *
from authgenforge.data.forensics_dataset import _compute_edge_mask


class ForensicsMDSDataset(Dataset):
    """
    MDS-shard backed sibling of ForensicsDataset. Reads raw image + mask
    bytes out of MosaicML Streaming (MDS) shards built by
    packages/mdsconverter/build_mds_dataset.py instead of loose files —
    same transforms and the same {image, mask, edge_mask} return dict.

    Authentic samples are stored with empty mask bytes; an all-zero mask
    the size of the image is built here instead.

    mds_root accepts one MDS split directory (e.g. <out>/train) or a
    list of them, concatenated in order — the same list-of-roots
    convention as ForensicsDataset.

    Reads shards with streaming's per-shard readers (one MDSReader per
    shard, index -> shard via bisect over cumulative sample counts)
    rather than StreamingDataset. The shards are fully local, so
    StreamingDataset's download/eviction machinery buys nothing — and its
    shared-memory coordination expects a "local leader" process to have
    registered first, which DataLoader workers on Windows (spawned, not
    forked) never find: they fail with "shared memory prefix was not
    registered by local leader". An MDSReader just opens the shard file
    per read, so it pickles into spawned workers and is safe to share.

    Map-style, so plain DataLoader / StatefulDataLoader handle
    duplicate-free multi-worker sharding and resume.
    """

    def __init__(
        self,
        mds_root: str | list[str],
        transform,
        edge_kernel_size: int = 7,
    ):
        from streaming.base.format import reader_from_json

        roots = (
            [mds_root]
            if isinstance(mds_root, str)
            else list(mds_root)
        )

        if not roots:
            raise ValueError(
                "mds_root must be a non-empty path or list of paths"
            )

        self.readers = []
        self.offsets = []  # global index of each shard's first sample

        total = 0

        for root in roots:

            for shard in _load_index(root)["shards"]:

                self.readers.append(
                    reader_from_json(root, None, shard)
                )
                self.offsets.append(total)

                total += shard["samples"]

        if total == 0:
            raise ValueError(
                f"No samples found in MDS root(s) {roots}"
            )

        self._length = total
        self.transform = transform
        self.edge_kernel_size = edge_kernel_size

    def __len__(self) -> int:
        return self._length

    def get_raw(self, index: int) -> dict:
        """
        The stored sample dict (bytes + metadata columns), undecoded.
        """

        if not 0 <= index < self._length:
            raise IndexError(
                f"index {index} out of range for {self._length} samples"
            )

        shard = bisect.bisect_right(self.offsets, index) - 1

        return self.readers[shard].get_item(
            index - self.offsets[shard]
        )

    def __getitem__(self, index: int) -> dict:

        sample = self.get_raw(index)

        image = Image.open(
            io.BytesIO(sample["image"])
        ).convert("RGB")

        if len(sample["mask"]) > 0:
            mask = Image.open(
                io.BytesIO(sample["mask"])
            ).convert("L")
        else:
            mask = Image.new("L", image.size, 0)

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


def _load_index(root: str) -> dict:

    index_path = os.path.join(root, "index.json")

    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"No MDS index.json under {root} — build it with "
            f"packages/mdsconverter/build_mds_dataset.py"
        )

    with open(index_path, encoding="utf-8") as f:
        return json.load(f)
