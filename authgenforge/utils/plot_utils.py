from __future__ import annotations

from authgenforge import *


LABEL_NAMES = {
    0: "Authentic",
    1: "Tampered",
}


def show_image(
    tensor: torch.Tensor,
    label: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Display a single image tensor.

    tensor : (C, H, W) or (1, C, H, W)
    label  : optional title
    ax     : existing Axes; creates one if None
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    img = (
        tensor.cpu()
        .float()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))

    ax.imshow(img)

    if label:
        ax.set_title(str(label), fontsize=9)

    ax.axis("off")

    return ax


def show_batch(
    batch,
    label: str = "",
    max_images: int | None = None,
) -> None:
    """
    Display a grid of images from a dataloader batch.

    Supports:
        {"image": ..., "labels": ...}

    or:
        (images, labels)
    """
    if isinstance(batch, dict):
        images = batch["image"]
        labels = batch.get("labels")
    else:
        images, labels = batch

    n = len(images) if max_images is None else min(len(images), max_images)

    if n == 0:
        return

    cols = min(8, n)
    rows = max(1, (n + cols - 1) // cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 2.2, rows * 2.4),
    )

    axes = np.array(axes).flatten()

    for i in range(n):
        img_label = ""

        if labels is not None:
            raw = labels[i]
            cls_idx = raw.item() if hasattr(raw, "item") else int(raw)
            img_label = LABEL_NAMES.get(cls_idx, str(cls_idx))

        show_image(
            images[i],
            label=img_label,
            ax=axes[i],
        )

    for ax in axes[n:]:
        ax.axis("off")

    if label:
        fig.suptitle(
            label,
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show()