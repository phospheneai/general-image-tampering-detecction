from __future__ import annotations

from authgenforge import *


def _get_layer_id(name: str) -> int:
    """
    Extract the DINOv3 transformer block index from a parameter name.

    The Hugging Face transformers DINOv3 implementation (loaded via
    AutoModel, then wrapped by peft's get_peft_model) names parameters
    like:

        backbone.dino.base_model.model.layer.0.attention.q_proj.lora_A.default.weight
        backbone.dino.base_model.model.layer.1.attention.k_proj.lora_B.default.weight
        ...

    i.e. "layer.N", not "blocks.N" — that naming belongs to the native/
    timm-style DINOv3 checkpoint format, which this project does not use
    (verified empirically against the installed transformers/peft
    versions; matching on "blocks." here would silently match nothing,
    making every parameter fall back to layer 0 with no error).

    Parameters that do not belong to a transformer block are assigned
    layer 0.
    """

    parts = name.split(".")

    for i, part in enumerate(parts):
        if part == "layer" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except (TypeError, ValueError):
                pass

    return 0


def get_optimizer(
    model: nn.Module,
    base_lr: float = 5e-4,
    weight_decay: float = 0.05,
    layer_decay: float = 0.8,
):
    """
    Build AdamW optimizer for DINOv3 + LoRA segmentation.

    Only parameters with requires_grad=True are included.

    LoRA parameters inside DINOv3 receive layer-wise learning-rate
    decay based on their transformer block.

    Segmentation-head parameters use the base learning rate.

    Parameters without weight decay:
        - 1D parameters
        - bias parameters
        - normalization parameters
    """

    # ----------------------------------------------------------
    # Determine the number of DINOv3 transformer blocks
    # ----------------------------------------------------------

    block_ids = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if "backbone.dino." in name and ".layer." in name:
            block_id = _get_layer_id(name)
            block_ids.append(block_id)

    num_layers = (
        max(block_ids) + 1
        if block_ids
        else 1
    )

    # Earlier transformer blocks receive smaller LR.
    layer_scales = {
        layer_id:
        layer_decay ** (num_layers - layer_id - 1)
        for layer_id in range(num_layers)
    }

    # ----------------------------------------------------------
    # Parameter groups
    # ----------------------------------------------------------

    param_groups = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        # ------------------------------------------------------
        # Learning rate
        # ------------------------------------------------------

        if (
            "backbone.dino." in name
            and ".layer." in name
            and "lora_" in name
        ):

            layer_id = _get_layer_id(name)

            lr = (
                base_lr
                * layer_scales.get(
                    layer_id,
                    1.0,
                )
            )

        else:

            # Segmentation head and any other trainable
            # parameters use the base learning rate.
            lr = base_lr

        # ------------------------------------------------------
        # Weight decay
        # ------------------------------------------------------

        no_decay = (
            param.ndim == 1
            or name.endswith(".bias")
            or "norm" in name.lower()
        )

        param_groups.append(
            {
                "params": [param],
                "lr": lr,
                "weight_decay": (
                    0.0
                    if no_decay
                    else weight_decay
                ),
            }
        )

    # ----------------------------------------------------------
    # AdamW
    # ----------------------------------------------------------

    return torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )