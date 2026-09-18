from __future__ import annotations

from authgenforge import *


class PixelBCEWithLogitsLoss(nn.Module):
    """
    Pixel-wise Binary Cross Entropy loss for forgery segmentation.

    The model produces one logit per pixel, and the loss compares
    those logits against the binary ground-truth forgery mask.
    """

    def __init__(self):
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        return self.loss(logits, mask)