"""
Shard-local shuffling for MDS splits read from a mounted S3 prefix.

The default train loader shuffles over the whole split (shuffle=True), so
consecutive samples come from random shards. That is fine when every shard
is on a local disk, but when the split is a mounted S3 prefix
(authgenforge/utils/s3_mount.py; processed-v1 train is ~1.1 TB in ~535 MB
shards), every sample would be a read from a different, far-away file.

ShardBlockSampler instead:

  1. shuffles the shard order (new order every epoch),
  2. groups it into blocks of `block_shards` shards,
  3. yields the samples of one block in random order, then the next block.

So reads stay within `block_shards` files at a time and every shard is
visited once per epoch. Mixing is across `block_shards` random shards at a
time (the split's shards are already class-interleaved by
build_mds_dataset.py).

Stateful (state_dict / load_state_dict), so torchdata's StatefulDataLoader
resumes mid-epoch at the exact position: the order is a pure function of
(seed, epoch), and the state records the epoch and how many indices were
yielded.

Opt-in via datasets.train.shard_block in the training yml; unset keeps
the plain shuffle.
"""

from __future__ import annotations

import threading

import numpy as np
from torch.utils.data import Sampler


class ShardBlockSampler(Sampler[int]):

    def __init__(
        self,
        samples_per_shard,
        block_shards: int,
        seed: int = 0,
        prefetch=None,
    ):
        """
        samples_per_shard: sample count of each shard, in the dataset's
            global index order (StreamingDataset.samples_per_shard).
        block_shards: shards mixed together at a time.
        prefetch: optional callable(shard_id) that gets a shard ready;
            called from a background thread for the next block.
        """

        if block_shards < 1:
            raise ValueError(
                f"block_shards must be >= 1, got {block_shards}"
            )

        self.samples_per_shard = np.asarray(
            samples_per_shard,
            dtype=np.int64,
        )
        self.shard_offsets = np.concatenate(
            [[0], np.cumsum(self.samples_per_shard)]
        )
        self.block_shards = int(block_shards)
        self.seed = int(seed)
        self.prefetch = prefetch

        self.epoch = 0
        self._yielded = 0
        self._skip = 0

    def __len__(self) -> int:
        return int(self.shard_offsets[-1])

    # ----------------------------------------------------------
    # Order
    # ----------------------------------------------------------

    def _blocks(self, epoch: int) -> list[np.ndarray]:
        """
        Shard-id blocks for this epoch.
        """

        rng = np.random.default_rng(
            [self.seed, epoch]
        )
        shard_order = rng.permutation(
            len(self.samples_per_shard)
        )

        return [
            shard_order[i:i + self.block_shards]
            for i in range(
                0,
                len(shard_order),
                self.block_shards,
            )
        ]

    def _block_indices(self, epoch: int, block_id: int, shards) -> np.ndarray:
        indices = np.concatenate(
            [
                np.arange(
                    self.shard_offsets[s],
                    self.shard_offsets[s + 1],
                )
                for s in shards
            ]
        )
        rng = np.random.default_rng(
            [self.seed, epoch, block_id + 1]
        )
        return rng.permutation(indices)

    def _prefetch_async(self, shards) -> None:
        if self.prefetch is None:
            return

        def run():
            for s in shards:
                try:
                    self.prefetch(int(s))
                except Exception as e:  # a failed prefetch is retried on read
                    print(
                        f"[ShardBlockSampler] prefetch of shard {s} "
                        f"failed: {e}",
                        flush=True,
                    )

        threading.Thread(
            target=run,
            daemon=True,
        ).start()

    def __iter__(self):
        epoch = self.epoch
        skip = self._skip
        self._skip = 0
        self._yielded = skip

        blocks = self._blocks(epoch)
        position = 0
        started = False

        for block_id, shards in enumerate(blocks):

            block_len = int(
                self.samples_per_shard[shards].sum()
            )

            if position + block_len <= skip:
                position += block_len
                continue

            if not started:
                # nothing prefetched this one: fresh epoch or a resume
                self._prefetch_async(shards)
                started = True

            if block_id + 1 < len(blocks):
                self._prefetch_async(
                    blocks[block_id + 1]
                )

            indices = self._block_indices(
                epoch,
                block_id,
                shards,
            )
            start = max(0, skip - position)
            position += block_len

            for index in indices[start:]:
                self._yielded += 1
                yield int(index)

        self.epoch = epoch + 1
        self._yielded = 0

    # ----------------------------------------------------------
    # torchdata Stateful protocol
    # ----------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "yielded": self._yielded,
        }

    def load_state_dict(self, state: dict) -> None:
        self.epoch = int(state["epoch"])
        self._skip = int(state["yielded"])
