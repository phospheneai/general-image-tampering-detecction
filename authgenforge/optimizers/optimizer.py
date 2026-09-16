from __future__ import annotations

from authgenforge import *


def get_optimizer(
    model: nn.Module,
    base_lr: float = 5e-4,
    weight_decay: float = 0.05,
    layer_decay: float = 0.8,
):
    """
    Build an AdamW optimizer with optional layer-wise learning-rate decay.

    Parameters
    ----------
    model:
        Model whose trainable parameters will be optimized.

    base_lr:
        Learning rate for the highest/default layer.

    weight_decay:
        AdamW weight decay applied to parameters that are not biases/norms.

    layer_decay:
        Multiplicative learning-rate decay between backbone layers.
    """

    num_layers = len(list(model.children()))

    layer_scales = {
        i: layer_decay ** (num_layers - i - 1)
        for i in range(num_layers)
    }

    def get_layer_id(name: str) -> int:
        """
        Determine a parameter's layer ID.

        This currently recognizes common encoder-style parameter names.
        It can be adapted once the final forgery model architecture
        is decided.
        """
        if "encoder" in name:
            parts = name.split(".")

            try:
                return int(parts[2])
            except (IndexError, ValueError):
                return 0

        return 0

    param_groups = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        no_decay = (
            param.ndim == 1
            or name.endswith(".bias")
            or "norm" in name.lower()
        )

        layer_id = get_layer_id(name)

        param_groups.append(
            {
                "params": [param],
                "lr": base_lr * layer_scales.get(layer_id, 1.0),
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        )

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    return optimizer