from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from authgenforge import *
from authgenforge.augmentations.presets import (
    get_train_transforms,
    get_val_transforms,
)
from authgenforge.utils.aug_utils import recursively_read


class ForensicsDataset(Dataset):
    """
    Folder-based forgery segmentation dataset.

    Expected layout, per root directory:

        <root>/images/<name>.<ext>
        <root>/masks/<name>.png

    Every image must have a matching mask with the same filename stem.
    Authentic (pristine) images use an all-zero mask; tampered images
    use a binary {0, 255} mask marking the forged region. This matches
    the CASIA-style layout used throughout DATASETS.md.
    """

    def __init__(
        self,
        root_dirs: str | list[str],
        transform,
        edge_kernel_size: int = 7,
    ):
        roots = (
            [root_dirs]
            if isinstance(root_dirs, str)
            else list(root_dirs)
        )

        self.samples: list[tuple[str, str]] = []

        for root in roots:

            root = Path(root)
            image_dir = root / "images"
            mask_dir = root / "masks"

            if not image_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected an 'images' subdirectory under {root}"
                )

            for image_path in recursively_read(image_dir):

                stem = Path(image_path).stem
                mask_path = mask_dir / f"{stem}.png"

                if not mask_path.is_file():
                    raise FileNotFoundError(
                        f"No mask found for {image_path} "
                        f"(expected {mask_path})"
                    )

                self.samples.append(
                    (image_path, str(mask_path))
                )

        if not self.samples:
            raise ValueError(
                f"No image/mask pairs found under {roots}"
            )

        self.transform = transform
        self.edge_kernel_size = edge_kernel_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:

        image_path, mask_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

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


def _compute_edge_mask(
    mask: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """
    Derive a boundary-band weight map from a binary mask tensor.

    Computed as (dilate - erode) via max-pooling on the mask itself, so
    it always reflects the mask's final, post-augmentation geometry
    rather than a stale pre-crop/pre-flip version. All-zero for
    authentic images, since there is no forged region to have a
    boundary around.
    """

    batched = mask.unsqueeze(0)

    dilated = F.max_pool2d(
        batched,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    eroded = -F.max_pool2d(
        -batched,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    edge = (dilated - eroded).clamp(0.0, 1.0)

    return edge.squeeze(0)


def build_forensics_datasets(
    train_dir: str | list[str],
    test_dir: str | list[str],
    crop_size: int = 512,
    buffer_size: int = 1000,
    data_context: str = "normal",
    data_format: str = "folder",
) -> tuple[Dataset, Dataset]:
    """
    Build the train/test forgery segmentation datasets.

    data_format selects the backend for both splits:
        folder -> ForensicsDataset, train_dir/test_dir are
                  images/ + masks/ folders
        mds    -> ForensicsMDSDataset, train_dir/test_dir are MDS
                  split dirs written by
                  packages/mdsconverter/build_mds_dataset.py

    buffer_size and data_context are accepted for interface parity with
    the streaming, multi-domain pipeline (see sagemaker/README.md's
    "Current status" section) — neither backend uses them yet.
    """

    if data_format == "folder":
        dataset_cls = ForensicsDataset
    elif data_format == "mds":
        from authgenforge.data.forensics_mds_dataset import (
            ForensicsMDSDataset,
        )
        dataset_cls = ForensicsMDSDataset
    else:
        raise ValueError(
            f"data_format must be 'folder' or 'mds', got {data_format!r}"
        )

    train_ds = dataset_cls(
        train_dir,
        transform=get_train_transforms(
            crop_size=crop_size
        ),
    )

    test_ds = dataset_cls(
        test_dir,
        transform=get_val_transforms(
            crop_size=crop_size
        ),
    )

    return train_ds, test_ds
