"""
Sanity check: overfit the model on a single frozen batch.
Loss should converge toward zero within a few hundred iterations.
If it can't, that's an architecture/loss wiring bug — check this before
spending GPU time debugging a full run that isn't converging.
Run from the project root:
    python tests/overfit_single_batch.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from authgenforge.options.load import (
    get_criterion_from_yml,
    get_model_from_yml,
    get_sample_from_yml,
)

CONFIG_PATH = "configs/normal/train_forensics.yml"
N_ITERS     = 200
LR          = 5e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# Forward
# ------------------------------------------------------------------

def _forward(model, images, masks, edge_masks, criterion):
    # bf16 autocast — matches SegmentationTrainer.train()'s mixed_precision
    # path (see authgenforge/training/trainer.py); without it this is a
    # plain fp32 forward/backward through the full DINOv3 backbone, which
    # OOMs at the config's real batch_size/crop_size on a small GPU. bf16
    # (not fp16) needs no GradScaler — no underflow risk, so backward()/
    # step() below stay plain.
    with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(images)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        result = criterion(logits=logits, mask=masks, edge_mask=edge_masks)
        loss, breakdown = result if isinstance(result, (tuple, list)) else (result, {})

    return loss, logits, breakdown


# ------------------------------------------------------------------
# Overfit loop
# ------------------------------------------------------------------

def overfit_on_single_batch(model, batch, criterion, optimizer, n_iters=N_ITERS):
    model.train()
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    edge_masks = batch.get("edge_mask")
    if edge_masks is not None:
        edge_masks = edge_masks.to(device)

    pbar = tqdm(range(n_iters), desc="Overfit")
    for _ in pbar:
        optimizer.zero_grad()
        loss, logits, breakdown = _forward(model, images, masks, edge_masks, criterion)
        loss.backward()
        optimizer.step()

        extra = {k: f"{v:.4f}" for k, v in breakdown.items()}
        pbar.set_postfix({"loss": f"{loss.item():.6f}", **extra})

        del logits


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Device: {device}")

    model     = get_model_from_yml(CONFIG_PATH).to(device)
    criterion = get_criterion_from_yml(CONFIG_PATH)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)

    sample = get_sample_from_yml(CONFIG_PATH)
    print(f"Sample — image: {sample['image'].shape}  mask: {sample['mask'].shape}")

    overfit_on_single_batch(model, sample, criterion, optimizer)
