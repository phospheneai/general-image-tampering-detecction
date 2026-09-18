from __future__ import annotations

from authgenforge import *


class EdgeWeightedBCEWithLogitsLoss(nn.Module):
    """
    Edge-weighted Binary Cross Entropy loss for forgery segmentation.

    The edge mask provides higher importance to pixels around the
    boundaries of manipulated regions.

    The final edge loss is scaled by edge_lambda.
    """

    def __init__(self, edge_lambda: float = 20.0):
        super().__init__()

        self.edge_lambda = edge_lambda

    def forward(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:

        loss = F.binary_cross_entropy_with_logits(
            input=logits,
            target=mask,
            weight=edge_mask,
        )

        return self.edge_lambda * loss