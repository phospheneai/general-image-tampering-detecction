from __future__ import annotations

from authgenforge import *

from authgenforge.utils.logger import get_logger
from authgenforge.utils.meters import AverageMeter

import gc


# ============================================================
# Segmentation metrics
# ============================================================


class _SegmentationMetricsAccumulator:
    """
    Accumulates pixel-level segmentation statistics without
    storing every prediction and target tensor.

    Metrics:
        - IoU
        - F1
        - precision
        - recall
        - accuracy
    """

    __slots__ = (
        "tp",
        "tn",
        "fp",
        "fn",
    )

    def __init__(self) -> None:
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:

        predictions = (
            predictions.detach()
            .reshape(-1)
        )

        targets = (
            targets.detach()
            .reshape(-1)
        )

        predictions = predictions >= 0.5
        targets = targets >= 0.5

        self.tp += int(
            torch.count_nonzero(
                predictions & targets
            ).item()
        )

        self.tn += int(
            torch.count_nonzero(
                (~predictions) & (~targets)
            ).item()
        )

        self.fp += int(
            torch.count_nonzero(
                predictions & (~targets)
            ).item()
        )

        self.fn += int(
            torch.count_nonzero(
                (~predictions) & targets
            ).item()
        )

    def metrics(
        self,
        loss: float,
    ) -> dict:

        total = (
            self.tp
            + self.tn
            + self.fp
            + self.fn
        )

        if total == 0:
            return {
                "loss": float(loss),
                "accuracy": 0.0,
                "iou": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

        precision = (
            self.tp
            / (self.tp + self.fp)
            if (self.tp + self.fp) > 0
            else 0.0
        )

        recall = (
            self.tp
            / (self.tp + self.fn)
            if (self.tp + self.fn) > 0
            else 0.0
        )

        iou = (
            self.tp
            / (
                self.tp
                + self.fp
                + self.fn
            )
            if (
                self.tp
                + self.fp
                + self.fn
            ) > 0
            else 0.0
        )

        f1_denominator = (
            precision + recall
        )

        f1 = (
            2.0
            * precision
            * recall
            / f1_denominator
            if f1_denominator > 0
            else 0.0
        )

        accuracy = (
            self.tp + self.tn
        ) / total

        return {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "iou": float(iou),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }

    def confusion_matrix(
        self,
    ) -> np.ndarray:

        return np.array(
            [
                [self.tn, self.fp],
                [self.fn, self.tp],
            ],
            dtype=np.int64,
        )

    def clear(self) -> None:

        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0


# ============================================================
# Trainer
# ============================================================


class SegmentationTrainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        test_loader,
        criterion: nn.Module,
        optimizer,
        scheduler=None,
        device: str | None = None,
        experiment_name: str = "experiment",
        save_checkpoint_dir: str = "checkpoints",
        mixed_precision: bool = True,
        grad_accum_steps: int = 1,
        grad_clip: float = 1.0,
        log_interval: int = 100,
        save_interval: int = 2000,
        best_iou: float = 0.0,
        load_checkpoint_path: str | None = None,
        logger: logging.Logger | None = None,
        resume_dataloader: bool = True,
    ):

        # --------------------------------------------------------
        # Basic configuration
        # --------------------------------------------------------

        # Auto-detect rather than hard-defaulting to "cuda" - a
        # GPU-only default crashes outright on a CPU-only machine
        # (e.g. local development before deploying to a GPU box).
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device

        self.use_cuda = (
            str(device).startswith("cuda")
            and torch.cuda.is_available()
        )

        self.autocast_device = (
            str(device).split(":", 1)[0]
        )

        self.model = model.to(
            device
        )

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.criterion = criterion.to(
            device
        )

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.experiment_name = (
            experiment_name
        )

        # Mixed precision is meaningful here only for
        # CUDA-based SageMaker training.
        self.mixed_precision = (
            mixed_precision
            and self.use_cuda
        )

        self.grad_accum_steps = max(
            1,
            int(grad_accum_steps),
        )

        self.grad_clip = grad_clip

        self.log_interval = max(
            1,
            int(log_interval),
        )

        self.save_interval = max(
            1,
            int(save_interval),
        )

        self.best_iou = best_iou

        # global_step = optimizer updates
        self.global_step = 0

        # epoch_start = first epoch to execute
        self.epoch_start = 0

        # Mid-epoch resume: the train loader position saved with a
        # periodic checkpoint (StatefulDataLoader only). Applied once,
        # to the first epoch train() runs after load_checkpoint().
        self.resume_dataloader = resume_dataloader
        self._resume_loader_state = None
        self._resume_batch = 0

        # --------------------------------------------------------
        # Mixed precision
        # --------------------------------------------------------

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.mixed_precision,
        )

        # --------------------------------------------------------
        # Checkpoint directories
        # --------------------------------------------------------

        self.ckpt_dir = (
            Path(save_checkpoint_dir)
            / experiment_name
        )

        self.ckpt_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        run_tag = (
            datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S"
            )
        )

        self.run_dir = (
            self.ckpt_dir / run_tag
        )

        self.log_dir = (
            self.run_dir / "logs"
        )

        self.plot_dir = (
            self.run_dir / "plots"
        )

        self.prediction_dir = (
            self.run_dir / "predictions"
        )

        for directory in (
            self.log_dir,
            self.plot_dir,
            self.prediction_dir,
        ):

            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        self.log_txt = (
            self.log_dir / "log.txt"
        )

        # One row per epoch. Lives next to the checkpoints, not in the
        # dated run folder, so a resumed job keeps appending to one file.
        self.metrics_csv = (
            self.ckpt_dir / "metrics.csv"
        )

        self.log = logger or get_logger(
            name=experiment_name,
            log_dir=str(
                self.log_dir
            ),
        )

        if self.use_cuda:
            torch.backends.cudnn.benchmark = True

        # --------------------------------------------------------
        # Resume
        # --------------------------------------------------------

        if load_checkpoint_path is not None:

            self.load_checkpoint(
                load_checkpoint_path
            )

    # ==========================================================
    # Memory
    # ==========================================================

    def _clear_memory(self) -> None:

        gc.collect()

        if self.use_cuda:
            torch.cuda.empty_cache()

    # ==========================================================
    # Current LR
    # ==========================================================

    def _current_lr(self) -> float:

        # Our custom CosineLRScheduler exposes .lr.
        if (
            self.scheduler is not None
            and hasattr(
                self.scheduler,
                "lr",
            )
        ):

            return float(
                self.scheduler.lr
            )

        # Fallback for torch schedulers.
        if (
            self.scheduler is not None
            and hasattr(
                self.scheduler,
                "get_last_lr",
            )
        ):

            lrs = (
                self.scheduler
                .get_last_lr()
            )

            if lrs:
                return float(
                    lrs[0]
                )

        if getattr(
            self.optimizer,
            "param_groups",
            None,
        ):

            return float(
                self.optimizer
                .param_groups[0]
                .get(
                    "lr",
                    0.0,
                )
            )

        return 0.0

    # ==========================================================
    # Logging
    # ==========================================================

    def _log(
        self,
        message: str,
    ) -> None:

        self.log.info(
            message
        )

        with open(
            self.log_txt,
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                message + "\n"
            )

    # ==========================================================
    # Result files
    # ==========================================================

    METRIC_NAMES = (
        "loss",
        "iou",
        "f1",
        "precision",
        "recall",
        "accuracy",
    )

    PREDICTION_FIELDS = (
        "image_name",
        "ground_truth",
        "probability",
        "predicted_class",
        "iou",
    )

    def _append_epoch_metrics(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
    ) -> None:
        """
        Append this epoch's train/val metrics to metrics.csv.
        """

        fieldnames = ["epoch"] + [
            f"{phase}_{name}"
            for phase in ("train", "val")
            for name in self.METRIC_NAMES
        ]

        row = {"epoch": epoch}

        for phase, metrics in (
            ("train", train_metrics),
            ("val", val_metrics),
        ):
            for name in self.METRIC_NAMES:
                row[f"{phase}_{name}"] = (
                    f"{metrics.get(name, 0):.6f}"
                )

        is_new = not self.metrics_csv.is_file()

        with open(
            self.metrics_csv,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            if is_new:
                writer.writeheader()

            writer.writerow(row)

    def _prediction_rows(
        self,
        batch: dict,
        probabilities: torch.Tensor,
        masks: torch.Tensor,
        first_index: int,
    ) -> list[dict]:
        """
        One row per validation image.

        ground_truth     image-level class, 0 authentic / 1 tampered
        probability      highest tampered-pixel probability in the image
        predicted_class  1 if any pixel is predicted tampered (>= 0.5)
        iou              predicted mask vs ground-truth mask for this image;
                         1.0 when both are empty
        """

        batch_size = probabilities.size(0)

        probabilities = (
            probabilities.detach()
            .float()
            .reshape(batch_size, -1)
        )

        targets = (
            masks.detach()
            .reshape(batch_size, -1)
            >= 0.5
        )

        predictions = probabilities >= 0.5

        intersection = (
            (predictions & targets)
            .sum(dim=1)
            .float()
        )

        union = (
            (predictions | targets)
            .sum(dim=1)
            .float()
        )

        iou = torch.where(
            union > 0,
            intersection / union.clamp(min=1),
            torch.ones_like(union),
        )

        image_probability = (
            probabilities.max(dim=1).values
        )

        labels = batch.get("label")

        if labels is None:
            labels = targets.any(dim=1)

        names = batch.get("name")

        if names is None:
            names = [
                f"sample_{first_index + i:07d}"
                for i in range(batch_size)
            ]

        return [
            {
                "image_name": names[i],
                "ground_truth": int(labels[i]),
                "probability": (
                    f"{image_probability[i].item():.6f}"
                ),
                "predicted_class": int(
                    image_probability[i].item() >= 0.5
                ),
                "iou": f"{iou[i].item():.6f}",
            }
            for i in range(batch_size)
        ]

    # ==========================================================
    # Checkpointing
    # ==========================================================

    def save_checkpoint(
        self,
        state: dict,
        is_best: bool = False,
    ) -> None:

        state = dict(state)

        state.update(
            {
                "model_state_dict":
                    self.model.state_dict(),

                "optimizer_state_dict":
                    self.optimizer.state_dict(),

                "scheduler_state_dict":
                    (
                        self.scheduler.state_dict()
                        if self.scheduler is not None
                        else None
                    ),

                "scaler_state_dict":
                    self.scaler.state_dict(),

                "global_step":
                    self.global_step,

                "best_iou":
                    self.best_iou,
            }
        )

        path = (
            self.ckpt_dir
            / "latest_checkpoint.pth"
        )

        tmp_path = path.with_suffix(
            ".tmp"
        )

        torch.save(
            state,
            tmp_path,
        )

        os.replace(
            tmp_path,
            path,
        )

        if is_best:

            best_path = (
                self.ckpt_dir
                / (
                    f"{self.experiment_name}"
                    "_best.pth"
                )
            )

            shutil.copyfile(
                path,
                best_path,
            )

            self._log(
                "Best model saved -> "
                f"{best_path} "
                f"(IoU={self.best_iou:.4f})"
            )

    def _save_epoch_snapshot(
        self,
        epoch: int,
    ) -> None:

        path = (
            self.ckpt_dir
            / f"epoch{epoch}.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "global_step":
                    self.global_step,
                "model_state_dict":
                    self.model.state_dict(),
                "best_iou":
                    self.best_iou,
            },
            path,
        )

        self._log(
            f"Epoch snapshot -> {path}"
        )

    def load_checkpoint(
        self,
        path: str,
    ) -> None:

        self._log(
            f"Loading checkpoint from {path}"
        )

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        if (
            "optimizer_state_dict"
            in checkpoint
        ):

            self.optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        if (
            self.scheduler is not None
            and checkpoint.get(
                "scheduler_state_dict"
            ) is not None
        ):

            self.scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        if (
            "scaler_state_dict"
            in checkpoint
        ):

            self.scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )

        self.best_iou = float(
            checkpoint.get(
                "best_iou",
                0.0,
            )
        )

        self.epoch_start = int(
            checkpoint.get(
                "epoch",
                0,
            )
        )

        self.global_step = int(
            checkpoint.get(
                "global_step",
                0,
            )
        )

        # Only periodic (mid-epoch) checkpoints carry a loader state;
        # epoch-end ones start the next epoch from batch 0 anyway.
        loader_state = checkpoint.get(
            "train_loader_state"
        )

        if (
            self.resume_dataloader
            and loader_state is not None
            and hasattr(
                self.train_loader,
                "load_state_dict",
            )
        ):

            self._resume_loader_state = loader_state

            self._resume_batch = int(
                checkpoint.get(
                    "step",
                    0,
                )
            )

        self._log(
            "Resumed - "
            f"epoch {self.epoch_start} "
            f"step {self.global_step} "
            f"best_iou "
            f"{self.best_iou:.4f}"
        )

    # ==========================================================
    # Model forward
    # ==========================================================

    def _model_output(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:

        output = self.model(
            images
        )

        if isinstance(
            output,
            (tuple, list),
        ):

            output = output[0]

        return output

    # ==========================================================
    # Criterion
    # ==========================================================

    def _criterion_forward(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
    ):

        """
        Expected criterion interface:

            loss, loss_breakdown = criterion(
                logits=logits,
                mask=mask,
                edge_mask=edge_mask,
            )

        A criterion returning only a scalar loss is also supported.
        """

        result = self.criterion(
            logits=logits,
            mask=mask,
            edge_mask=edge_mask,
        )

        if isinstance(
            result,
            (tuple, list),
        ):

            loss = result[0]

            loss_breakdown = (
                result[1]
                if len(result) > 1
                else {}
            )

        else:

            loss = result
            loss_breakdown = {}

        return (
            loss,
            loss_breakdown,
        )

    # ==========================================================
    # Metrics
    # ==========================================================

    def _print_metrics(
        self,
        phase: str,
        epoch: int,
        metrics: dict,
    ) -> None:

        message = (
            f"[{phase}] "
            f"Epoch {epoch} | "
            f"loss "
            f"{metrics.get('loss', 0):.4f} | "
            f"IoU "
            f"{metrics.get('iou', 0):.4f} | "
            f"F1 "
            f"{metrics.get('f1', 0):.4f} | "
            f"precision "
            f"{metrics.get('precision', 0):.4f} | "
            f"recall "
            f"{metrics.get('recall', 0):.4f} | "
            f"accuracy "
            f"{metrics.get('accuracy', 0):.4f}"
        )

        self._log(
            message
        )

    # ==========================================================
    # Optimizer update
    # ==========================================================

    def _optimizer_step(self) -> bool:
        """
        Perform one optimizer update and one scheduler update.

        Returns:
            True if an optimizer update was performed.
        """

        self.scaler.unscale_(
            self.optimizer
        )

        if self.grad_clip > 0:

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip,
            )

        old_scale = (
            self.scaler.get_scale()
        )

        self.scaler.step(
            self.optimizer
        )

        self.scaler.update()

        new_scale = (
            self.scaler.get_scale()
        )

        # If GradScaler reduced the scale, the optimizer step
        # was skipped because of an overflow.
        optimizer_updated = (
            not self.mixed_precision
            or new_scale >= old_scale
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        if (
            optimizer_updated
            and self.scheduler is not None
        ):

            # IMPORTANT:
            # Scheduler is step-based, so it advances exactly
            # once for every optimizer update.
            self.scheduler.step()

        if optimizer_updated:

            self.global_step += 1

        return optimizer_updated

    # ==========================================================
    # Training entry point
    # ==========================================================

    def train_model(
        self,
        end_epoch: int,
    ) -> None:

        for epoch in range(
            self.epoch_start,
            end_epoch,
        ):

            # --------------------------------------------------
            # Training
            # --------------------------------------------------

            train_metrics = self.train(
                epoch
            )

            self._print_metrics(
                "Train",
                epoch + 1,
                train_metrics,
            )

            # --------------------------------------------------
            # Epoch snapshot
            # --------------------------------------------------

            self._save_epoch_snapshot(
                epoch + 1
            )

            # --------------------------------------------------
            # Validation
            # --------------------------------------------------

            val_metrics = self.validate(
                epoch + 1
            )

            self._print_metrics(
                "Val",
                epoch + 1,
                val_metrics,
            )

            # --------------------------------------------------
            # Best model
            # --------------------------------------------------

            self._append_epoch_metrics(
                epoch + 1,
                train_metrics,
                val_metrics,
            )

            is_best = (
                val_metrics["iou"]
                > self.best_iou
            )

            if is_best:

                self.best_iou = (
                    val_metrics["iou"]
                )

            # --------------------------------------------------
            # Full checkpoint
            # --------------------------------------------------

            self.save_checkpoint(
                {
                    "epoch":
                        epoch + 1,
                },
                is_best=is_best,
            )

            self._clear_memory()

    # ==========================================================
    # Training
    # ==========================================================

    def train(
        self,
        epoch: int,
    ) -> dict:

        self.model.train()

        loss_meter = AverageMeter()

        metrics = (
            _SegmentationMetricsAccumulator()
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        # Number of successful backward passes currently
        # accumulated toward the next optimizer update.
        accumulation_count = 0

        # Mid-epoch resume: continue at the saved loader position
        # instead of replaying the epoch from batch 0.
        first_step = 1

        if self._resume_loader_state is not None:

            self.train_loader.load_state_dict(
                self._resume_loader_state
            )

            first_step = self._resume_batch + 1

            self._log(
                "Resuming train loader at batch "
                f"{first_step} of epoch {epoch + 1}"
            )

            self._resume_loader_state = None
            self._resume_batch = 0

        total_batches = len(
            self.train_loader
        )

        progress = tqdm(
            enumerate(
                self.train_loader,
                first_step,
            ),
            desc=(
                f"Epoch {epoch + 1} [Train]"
            ),
            initial=first_step - 1,
            total=total_batches,
        )

        for step, batch in progress:

            images = None
            masks = None
            edge_masks = None
            logits = None
            loss = None

            try:

                # --------------------------------------------------
                # Data
                # --------------------------------------------------

                images = batch[
                    "image"
                ].to(
                    self.device,
                    non_blocking=True,
                )

                masks = batch[
                    "mask"
                ].to(
                    self.device,
                    non_blocking=True,
                )

                edge_masks = batch.get(
                    "edge_mask"
                )

                if edge_masks is not None:

                    edge_masks = (
                        edge_masks.to(
                            self.device,
                            non_blocking=True,
                        )
                    )

                # --------------------------------------------------
                # Forward + loss
                # --------------------------------------------------

                with torch.amp.autocast(
                    device_type=(
                        self.autocast_device
                    ),
                    enabled=(
                        self.mixed_precision
                    ),
                ):

                    logits = (
                        self._model_output(
                            images
                        )
                    )

                    loss, loss_breakdown = (
                        self._criterion_forward(
                            logits=logits,
                            mask=masks,
                            edge_mask=edge_masks,
                        )
                    )

                    scaled_loss = (
                        loss
                        / self.grad_accum_steps
                    )

                # --------------------------------------------------
                # Backward
                # --------------------------------------------------

                self.scaler.scale(
                    scaled_loss
                ).backward()

                accumulation_count += 1

                # --------------------------------------------------
                # Optimizer boundary
                # --------------------------------------------------

                is_last_batch = (
                    step == total_batches
                )

                should_update = (
                    accumulation_count
                    >= self.grad_accum_steps
                    or is_last_batch
                )

                if should_update:

                    self._optimizer_step()

                    accumulation_count = 0

                # --------------------------------------------------
                # Loss bookkeeping
                # --------------------------------------------------

                actual_loss = float(
                    loss.detach().item()
                )

                loss_meter.update(
                    actual_loss,
                    images.size(0),
                )

                # --------------------------------------------------
                # Metrics
                # --------------------------------------------------

                with torch.no_grad():

                    probabilities = (
                        torch.sigmoid(
                            logits.detach()
                        )
                    )

                    metrics.update(
                        probabilities,
                        masks,
                    )

                # --------------------------------------------------
                # Progress bar
                # --------------------------------------------------

                if (
                    step
                    % self.log_interval
                    == 0
                ):

                    progress.set_postfix(
                        {
                            "loss":
                                f"{loss_meter.avg:.4f}",
                            "lr":
                                f"{self._current_lr():.2e}",
                            "step":
                                self.global_step,
                        }
                    )

                # --------------------------------------------------
                # Periodic checkpoint
                # --------------------------------------------------

                # Only on the batch that performed the update:
                # global_step is unchanged for the next
                # grad_accum_steps - 1 batches, which would
                # otherwise re-save the same checkpoint each time.
                if (
                    should_update
                    and self.global_step > 0
                    and self.global_step
                    % self.save_interval
                    == 0
                ):

                    state = {
                        "epoch": epoch,
                        "step": step,
                    }

                    if hasattr(
                        self.train_loader,
                        "state_dict",
                    ):
                        state[
                            "train_loader_state"
                        ] = self.train_loader.state_dict()

                    self.save_checkpoint(
                        state
                    )

                    self._log(
                        "Checkpoint saved - "
                        f"epoch {epoch} "
                        f"step {step} "
                        f"global_step "
                        f"{self.global_step}"
                    )

            except RuntimeError as error:

                if (
                    "out of memory"
                    in str(error).lower()
                ):

                    self._log(
                        "OOM at batch "
                        f"{step} - "
                        "skipping batch and "
                        "resetting accumulated gradients"
                    )

                    self.optimizer.zero_grad(
                        set_to_none=True
                    )

                    accumulation_count = 0

                    self._clear_memory()

                    continue

                raise

            finally:

                del (
                    images,
                    masks,
                    edge_masks,
                    logits,
                    loss,
                )

        # ----------------------------------------------------------
        # Safety: clear any gradients that somehow remain.
        # ----------------------------------------------------------

        self.optimizer.zero_grad(
            set_to_none=True
        )

        result = metrics.metrics(
            loss_meter.avg
        )

        metrics.clear()

        self._clear_memory()

        return result

    # ==========================================================
    # Validation
    # ==========================================================

    @torch.inference_mode()
    def validate(
        self,
        epoch: int,
    ) -> dict:

        self.model.eval()

        loss_meter = AverageMeter()

        metrics = (
            _SegmentationMetricsAccumulator()
        )

        progress = tqdm(
            self.test_loader,
            desc=(
                f"Epoch {epoch} [Val]"
            ),
        )

        predictions_csv = (
            self.prediction_dir
            / f"val_epoch_{epoch}.csv"
        )

        csv_file = open(
            predictions_csv,
            "w",
            newline="",
            encoding="utf-8",
        )

        writer = csv.DictWriter(
            csv_file,
            fieldnames=self.PREDICTION_FIELDS,
        )

        writer.writeheader()

        seen = 0

        for batch in progress:

            images = batch[
                "image"
            ].to(
                self.device,
                non_blocking=True,
            )

            masks = batch[
                "mask"
            ].to(
                self.device,
                non_blocking=True,
            )

            edge_masks = batch.get(
                "edge_mask"
            )

            if edge_masks is not None:

                edge_masks = (
                    edge_masks.to(
                        self.device,
                        non_blocking=True,
                    )
                )

            with torch.amp.autocast(
                device_type=(
                    self.autocast_device
                ),
                enabled=(
                    self.mixed_precision
                ),
            ):

                logits = (
                    self._model_output(
                        images
                    )
                )

                loss, loss_breakdown = (
                    self._criterion_forward(
                        logits=logits,
                        mask=masks,
                        edge_mask=edge_masks,
                    )
                )

            loss_meter.update(
                float(loss.item()),
                images.size(0),
            )

            probabilities = (
                torch.sigmoid(
                    logits
                )
            )

            metrics.update(
                probabilities,
                masks,
            )

            writer.writerows(
                self._prediction_rows(
                    batch,
                    probabilities,
                    masks,
                    seen,
                )
            )

            seen += images.size(0)

            progress.set_postfix(
                {
                    "loss":
                        f"{loss_meter.avg:.4f}",
                    "IoU":
                        f"{metrics.metrics(loss_meter.avg)['iou']:.4f}",
                }
            )

            del (
                images,
                masks,
                edge_masks,
                logits,
                probabilities,
                loss,
            )

        csv_file.close()

        self._log(
            f"Val predictions saved -> {predictions_csv}"
        )

        result = metrics.metrics(
            loss_meter.avg
        )

        metrics.clear()

        self._clear_memory()

        return result