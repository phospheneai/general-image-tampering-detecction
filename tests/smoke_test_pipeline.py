"""
Smoke test: exercises the full training pipeline for a few steps only.
Verifies imports, data loading, forward pass, backward pass,
checkpoint saving, logging, and inference — without running full epochs.

Run from the project root:
    python tests/smoke_test_pipeline.py           # CUDA
    python tests/smoke_test_pipeline.py --cpu     # CPU
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows terminals default to a legacy codepage (e.g. cp1252) that can't
# encode the checkmark/cross symbols below - force UTF-8 regardless of
# platform rather than falling back to plain ASCII everywhere.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import itertools

from authgenforge import *
from authgenforge.options.load import load_pipeline_from_yml

CONFIG_PATH   = "configs/normal/train_forensics.yml"
N_TRAIN_STEPS = 3
N_VAL_STEPS   = 2

PASS  = "\033[92m✓\033[0m"
FAIL  = "\033[91m✗\033[0m"
WARN  = "\033[93m⚠\033[0m"
TITLE = "\033[1m"
RESET = "\033[0m"

def check(label, fn):
    try:
        result = fn()
        print(f"  {PASS} {label}")
        return result
    except Exception:
        print(f"  {FAIL} {label}")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = parser.parse_args()

    # ── 1. Imports ────────────────────────────────────────────────────────────

    print(f"\n{TITLE}[1] Imports{RESET}")
    check("torch available", lambda: torch.zeros(1))

    cuda_available = torch.cuda.is_available()
    DEVICE = "cpu" if args.cpu else ("cuda" if cuda_available else "cpu")

    if args.cpu:
        print(f"  {WARN}  --cpu flag set, running on CPU")
    elif not cuda_available:
        print(f"  {WARN}  CUDA not available, falling back to CPU")
    else:
        cap = torch.cuda.get_device_capability()
        print(f"     GPU: {torch.cuda.get_device_name(0)}  (sm_{cap[0]}{cap[1]})")

    print(f"     torch {torch.__version__}  |  device: {DEVICE}")
    AMP_ENABLED = DEVICE == "cuda"

    # ── 2. Pipeline init ──────────────────────────────────────────────────────

    print(f"\n{TITLE}[2] Load pipeline{RESET}")

    def _load():
        tl, vl, m, tr = load_pipeline_from_yml(CONFIG_PATH, steps_per_epoch=N_TRAIN_STEPS)
        if DEVICE != tr.device:
            tr.device = DEVICE
            tr.model  = m.to(DEVICE)
            tr.scaler = torch.amp.GradScaler(enabled=False)
        return tl, vl, m, tr

    train_loader, test_loader, model, trainer = check("load_pipeline_from_yml", _load)

    trainer.log_interval  = 1
    trainer.save_interval = 999_999
    trainer.mixed_precision = AMP_ENABLED

    print(f"     experiment : {trainer.experiment_name}")
    print(f"     ckpt dir   : {trainer.ckpt_dir}")
    print(f"     log dir    : {trainer.log_dir}")

    # ── 3. Data loading ───────────────────────────────────────────────────────

    print(f"\n{TITLE}[3] Data loading{RESET}")
    train_batch = check("get train batch", lambda: next(iter(train_loader)))
    test_batch  = check("get test batch",  lambda: next(iter(test_loader)))
    check("train image shape", lambda: train_batch["image"].shape)
    check("train mask shape",  lambda: train_batch["mask"].shape)
    print(f"     train : image {train_batch['image'].shape}  mask {train_batch['mask'].shape}")
    print(f"     test  : image {test_batch['image'].shape}   mask {test_batch['mask'].shape}")

    # ── 4. Forward pass ───────────────────────────────────────────────────────

    print(f"\n{TITLE}[4] Forward pass{RESET}")
    model.train()

    def _fwd():
        images = train_batch["image"].to(DEVICE)
        masks = train_batch["mask"].to(DEVICE)
        edge_masks = train_batch.get("edge_mask")
        if edge_masks is not None:
            edge_masks = edge_masks.to(DEVICE)
        with torch.amp.autocast(DEVICE, enabled=AMP_ENABLED):
            logits = trainer._model_output(images)
            loss, breakdown = trainer._criterion_forward(logits, masks, edge_masks)
        return logits, loss, breakdown

    logits, loss, breakdown = check("forward + criterion", _fwd)
    check("loss is finite", lambda: torch.isfinite(loss))
    print(f"     logits {logits.shape}  loss {loss.item():.4f}  breakdown {breakdown}")

    # this forward pass's activation graph is never backward()'d, so it must
    # be dropped explicitly — otherwise it stays resident (logits/loss keep
    # the whole graph alive via .grad_fn) straight through section 5's own
    # forward+backward passes, and a model this size OOMs a 32GB GPU holding
    # two live graphs at once.
    del logits, loss, breakdown
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ── 5. Training steps (backward + optimizer) ─────────────────────────────

    print(f"\n{TITLE}[5] Training steps (N={N_TRAIN_STEPS}){RESET}")
    model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    loss_history = []

    for step, batch in enumerate(itertools.islice(train_loader, N_TRAIN_STEPS), 1):
        images = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)
        edge_masks = batch.get("edge_mask")
        if edge_masks is not None:
            edge_masks = edge_masks.to(DEVICE, non_blocking=True)
        trainer.global_step += 1

        with torch.amp.autocast(DEVICE, enabled=AMP_ENABLED):
            logits = trainer._model_output(images)
            loss, _ = trainer._criterion_forward(logits, masks, edge_masks)

        trainer.scaler.scale(loss).backward()
        trainer.scaler.unscale_(trainer.optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.grad_clip)
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.scheduler.step()

        loss_history.append(loss.item())
        trainer._log(f"[smoke] step {step}/{N_TRAIN_STEPS}  loss {loss.item():.6f}  lr {trainer.scheduler.lr:.2e}")
        print(f"     step {step}  loss={loss.item():.4f}  lr={trainer.scheduler.lr:.2e}")

    check("all losses finite", lambda: all(torch.isfinite(torch.tensor(l)) for l in loss_history))

    # ── 6. Checkpoint saving ──────────────────────────────────────────────────

    print(f"\n{TITLE}[6] Checkpoint saving{RESET}")
    check("save_checkpoint",      lambda: trainer.save_checkpoint({"epoch": 0}))
    check("_save_epoch_snapshot", lambda: trainer._save_epoch_snapshot(epoch=0))

    ckpt_files = os.listdir(trainer.ckpt_dir)
    print(f"     files in ckpt_dir: {ckpt_files}")

    # ── 7. Logging ────────────────────────────────────────────────────────────

    print(f"\n{TITLE}[7] Logging{RESET}")
    check("log.txt exists",    lambda: trainer.log_txt.exists())
    check("log.txt non-empty", lambda: trainer.log_txt.stat().st_size > 0)
    print(f"     log file : {trainer.log_txt}")
    print(f"     size     : {trainer.log_txt.stat().st_size} bytes")

    # ── 8. Validation / inference ─────────────────────────────────────────────
    # Plain batched forward pass — matches SegmentationTrainer.validate(),
    # sigmoid + fixed 0.5 threshold (see _SegmentationMetricsAccumulator).

    print(f"\n{TITLE}[8] Validation / inference{RESET}")
    model.eval()
    val_probs = []

    def _infer_batch(images):
        with torch.amp.autocast(DEVICE, enabled=AMP_ENABLED):
            logits = trainer._model_output(images.to(DEVICE))
        return torch.sigmoid(logits.float()).flatten().cpu().tolist()

    with torch.inference_mode():
        for batch in itertools.islice(test_loader, N_VAL_STEPS):
            probs = check("infer batch", lambda b=batch: _infer_batch(b["image"]))
            val_probs.extend(probs)

    check("val probs in [0,1]", lambda: all(0.0 <= p <= 1.0 for p in val_probs))

    # ── Summary ────────────────────────────────────────────────────────────────

    print(f"\n{TITLE}{'='*50}{RESET}")
    print(f"{TITLE}  SMOKE TEST PASSED{RESET}")
    print(f"{'='*50}")
    print(f"  device      : {DEVICE}")
    print(f"  train steps : {N_TRAIN_STEPS}  |  avg loss : {sum(loss_history)/len(loss_history):.4f}")
    print(f"  val pixels  : {len(val_probs)}")
    print(f"  log         : {trainer.log_txt}")
    print(f"  checkpoints : {trainer.ckpt_dir}")
    print()


if __name__ == "__main__":
    main()
