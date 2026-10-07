"""
Mid-epoch resume of the train loader: a run interrupted after k batches and
resumed from the saved loader state must see exactly the batches an
uninterrupted run sees — for the default shuffle and for ShardBlockSampler,
with worker processes (prefetch runs the sampler ahead of the consumer).
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import Dataset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from authgenforge.data.shard_block_sampler import ShardBlockSampler

StatefulDataLoader = pytest.importorskip(
    "torchdata.stateful_dataloader"
).StatefulDataLoader


class _Indices(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.tensor(i)


SHARDS = [7, 5, 9, 6, 8, 4, 10]          # 49 samples, uneven shards


def _loader(sampler_kind, num_workers):
    ds = _Indices(sum(SHARDS))
    kw = dict(batch_size=3, num_workers=num_workers, drop_last=True)
    if num_workers:
        kw.update(persistent_workers=True, prefetch_factor=4)
    if sampler_kind == "shuffle":
        torch.manual_seed(0)
        return StatefulDataLoader(ds, shuffle=True, **kw)
    return StatefulDataLoader(ds, sampler=ShardBlockSampler(SHARDS, 3, seed=1), **kw)


def _epoch(it_or_loader, n=None):
    out = []
    for i, b in enumerate(it_or_loader):
        if n is not None and i == n:
            break
        out.append(b.tolist())
    return out


@pytest.mark.parametrize("sampler_kind", ["shuffle", "shard_block"])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_mid_epoch_resume_matches_uninterrupted(sampler_kind, num_workers):
    ref = _loader(sampler_kind, num_workers)
    full = [_epoch(ref), _epoch(ref)]           # two epochs, uninterrupted

    k = 5
    a = _loader(sampler_kind, num_workers)
    it = iter(a)
    head = [next(it).tolist() for _ in range(k)]
    state = a.state_dict()
    del it, a

    b = _loader(sampler_kind, num_workers)
    b.load_state_dict(state)
    tail = _epoch(b)
    nxt = _epoch(b)

    assert head + tail == full[0]
    assert nxt == full[1]


def test_shard_block_sampler_covers_split_with_local_blocks():
    s = ShardBlockSampler(SHARDS, block_shards=3, seed=0)
    offsets = [0]
    for n in SHARDS:
        offsets.append(offsets[-1] + n)
    shard_of = {i: sid for sid in range(len(SHARDS))
                for i in range(offsets[sid], offsets[sid + 1])}

    e0, e1 = list(s), list(s)
    assert sorted(e0) == list(range(sum(SHARDS))) == sorted(e1)
    assert e0 != e1                               # reshuffled per epoch

    # samples come block by block: at most 3 distinct shards in each block
    pos = 0
    for block in s._blocks(0):
        n = sum(SHARDS[b] for b in block)
        assert {shard_of[i] for i in e0[pos:pos + n]} == set(block.tolist())
        pos += n


def test_shard_block_sampler_prefetches_next_block():
    fetched = []
    s = ShardBlockSampler(SHARDS, block_shards=3, seed=0, prefetch=fetched.append)
    list(s)
    import time
    time.sleep(0.2)
    blocks = s._blocks(0)
    expected = {int(x) for b in blocks for x in b}
    assert expected <= set(fetched)
