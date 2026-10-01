"""
Mid-epoch resume of the train loader: a run interrupted after k batches and
resumed from the saved loader state must see exactly the batches an
uninterrupted run sees, with worker processes too (prefetch runs the sampler
ahead of the consumer).
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    torch.manual_seed(0)
    return StatefulDataLoader(ds, shuffle=True, **kw)


def _epoch(it_or_loader, n=None):
    out = []
    for i, b in enumerate(it_or_loader):
        if n is not None and i == n:
            break
        out.append(b.tolist())
    return out


@pytest.mark.parametrize("sampler_kind", ["shuffle"])
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

