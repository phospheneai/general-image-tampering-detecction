from __future__ import annotations

from authgenforge import *
from authgenforge.data.forensics_dataset import (
    build_forensics_datasets,
)


def build_dataloaders(
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
    data_format: str = "folder",
) -> tuple[DataLoader, DataLoader]:

    test_num_workers = (
        num_workers
        if test_num_workers is None
        else test_num_workers
    )

    train_ds, test_ds = build_forensics_datasets(
        train_dir=train_dir,
        test_dir=test_dir,
        crop_size=crop_size,
        buffer_size=buffer_size,
        data_context=data_context,
        data_format=data_format,
    )

    # ----------------------------------------------------------
    # Stateful loader for SageMaker training resume
    # ----------------------------------------------------------

    if stateful:

        from torchdata.stateful_dataloader import (
            StatefulDataLoader,
        )

        train_loader_cls = StatefulDataLoader

    else:

        train_loader_cls = DataLoader

    # ----------------------------------------------------------
    # Training loader
    # ----------------------------------------------------------

    train_loader = train_loader_cls(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=(
            4
            if num_workers > 0
            else None
        ),
    )

    # ----------------------------------------------------------
    # Evaluation loader
    # ----------------------------------------------------------

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=test_num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            test_num_workers > 0
        ),
        prefetch_factor=(
            4
            if test_num_workers > 0
            else None
        ),
    )

    return train_loader, test_loader


if __name__ == "__main__":

    train_loader, test_loader = (
        build_dataloaders(
            train_dir="data/train",
            test_dir="data/test",
            crop_size=512,
            batch_size=4,
            num_workers=0,
            stateful=False,
        )
    )

    train_batch = next(
        iter(train_loader)
    )

    print(
        "[train]"
        f" image={train_batch['image'].shape}"
        f" mask={train_batch['mask'].shape}"
        f" edge_mask={train_batch['edge_mask'].shape}"
    )

    test_batch = next(
        iter(test_loader)
    )

    print(
        "[test]"
        f" image={test_batch['image'].shape}"
        f" mask={test_batch['mask'].shape}"
        f" edge_mask={test_batch['edge_mask'].shape}"
    )