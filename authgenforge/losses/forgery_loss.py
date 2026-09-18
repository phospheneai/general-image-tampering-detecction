from __future__ import annotations

from authgenforge import *


class ForgerySegmentationLoss(nn.Module):
    """
    Combined loss for forgery segmentation.

    Total loss:

        total_loss = pixel_bce + edge_bce

    The pixel BCE provides supervision over the complete
    forgery mask, while the edge-weighted BCE gives additional
    emphasis to manipulated-region boundaries.
    """

    def __init__(
        self,
        edge_lambda: float = 20.0,
    ):
        super().__init__()

        self.pixel_loss = (
            PixelBCEWithLogitsLoss()
        )

        self.edge_loss = (
            EdgeWeightedBCEWithLogitsLoss(
                edge_lambda=edge_lambda
            )
        )

    def forward(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
    ):

        pixel_bce = self.pixel_loss(
            logits=logits,
            mask=mask,
        )

        # If no edge mask is provided, use only pixel BCE.
        if edge_mask is None:

            total_loss = pixel_bce

            return (
                total_loss,
                {
                    "pixel_bce": float(
                        pixel_bce.detach().item()
                    ),
                    "edge_bce": 0.0,
                },
            )

        edge_bce = self.edge_loss(
            logits=logits,
            mask=mask,
            edge_mask=edge_mask,
        )

        total_loss = (
            pixel_bce
            + edge_bce
        )

        return (
            total_loss,
            {
                "pixel_bce": float(
                    pixel_bce.detach().item()
                ),
                "edge_bce": float(
                    edge_bce.detach().item()
                ),
                "total": float(
                    total_loss.detach().item()
                ),
            },
        )