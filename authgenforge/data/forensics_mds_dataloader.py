"""
MDS-backend DataLoader builder — wraps forensics_mds_dataset.py's
ForensicsMDSDataset. Mirrors ai-generated-video-detection's
authgenvideo/data/video_mds_dataloader.py. See authgenforge/options/load.py's
_DATASET_BACKENDS dict for how it gets selected via config
(data_format: mds) without touching calling code.

Accepts the same kwargs as dataloader.py's build_dataloaders (the folder
backend) plus cache_dir / cache_limit for s3:// dataroots, and returns
(train_loader, test_loader).

The train loader is a torchdata StatefulDataLoader by default
(stateful=True), so a resumed run picks up at the exact shuffle position.
Works unmodified here because the dataset is map-style — it only needs
__len__/__getitem__ + a Sampler, nothing MDS-specific.
"""

from __future__ import annotations

from authgenforge import *
from authgenforge.data.forensics_mds_dataset import (
    build_forensics_mds_datasets,
)


def build_mds_dataloaders(
    train_dir: str | list[str],
    test_dir: str | list[str],
    crop_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 4,
    test_num_workers: int | None = None,
    pin_memory: bool = True,
    buffer_size: int = 1000,
    stateful: bool = True,
    data_context: str = "normal",
    cache_dir: str | None = None,
    cache_limit: str | int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    buffer_size and data_context are accepted for interface parity with
    the folder backend — unused here.
    """

    test_num_workers = (
        num_workers
        if test_num_workers is None
        else test_num_workers
    )

    train_ds, test_ds = build_forensics_mds_datasets(
        train_dir=train_dir,
        test_dir=test_dir,
        crop_size=crop_size,
        cache_dir=cache_dir,
        cache_limit=cache_limit,
    )

    if stateful:
        from torchdata.stateful_dataloader import (
            StatefulDataLoader,
        )
        train_loader_cls = StatefulDataLoader
    else:
        train_loader_cls = DataLoader

    # No class-balanced sampler: it would need every sample's label up
    # front, and in MDS the label sits inside each row next to the image
    # bytes — a full-corpus read on every launch. build_mds_dataset.py's
    # authentic:tampered interleaving already keeps every shard at the
    # split's global ratio; correct any imbalance with loss weighting.
    train_loader = train_loader_cls(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=(
            4
            if num_workers > 0
            else None
        ),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=test_num_workers,
        pin_memory=pin_memory,
        persistent_workers=test_num_workers > 0,
        prefetch_factor=(
            4
            if test_num_workers > 0
            else None
        ),
    )

    return train_loader, test_loader


if __name__ == "__main__":

    train_loader, test_loader = build_mds_dataloaders(
        train_dir="/home/ubuntu/data/processed/processed-v1/train",
        test_dir="/home/ubuntu/data/processed/processed-v1/test",
        crop_size=512,
        batch_size=4,
        num_workers=2,
        stateful=False,
    )

    for name, loader in (("train", train_loader), ("test", test_loader)):
        batch = next(iter(loader))
        print(
            f"[{name}]"
            f" image={tuple(batch['image'].shape)}"
            f" mask={tuple(batch['mask'].shape)}"
            f" edge_mask={tuple(batch['edge_mask'].shape)}"
        )
