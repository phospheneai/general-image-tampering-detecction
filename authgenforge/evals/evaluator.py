from __future__ import annotations

from authgenforge import *


# ============================================================
# Segmentation metrics accumulator
# ============================================================


class _SegmentationMetricsAccumulator:
    """
    Accumulates pixel-level segmentation statistics over the
    complete evaluation dataset.

    Predictions and targets are converted to binary masks using
    a fixed threshold before TP/TN/FP/FN are accumulated.

    Metrics are calculated from the accumulated counts, so the
    evaluator does not need to keep every predicted mask in RAM.
    """

    def __init__(
        self,
        threshold: float = 0.5,
    ):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Args:
            predictions:
                Forgery probabilities with shape
                (B, 1, H, W).

            targets:
                Binary ground-truth masks with shape
                (B, 1, H, W).
        """

        predictions = (
            predictions.detach()
            .float()
            .cpu()
        )

        targets = (
            targets.detach()
            .float()
            .cpu()
        )

        predictions = (
            predictions >= self.threshold
        )

        targets = (
            targets >= 0.5
        )

        self.tp += int(
            (predictions & targets)
            .sum()
            .item()
        )

        self.tn += int(
            (~predictions & ~targets)
            .sum()
            .item()
        )

        self.fp += int(
            (predictions & ~targets)
            .sum()
            .item()
        )

        self.fn += int(
            (~predictions & targets)
            .sum()
            .item()
        )

    def metrics(self) -> dict:
        """
        Return accumulated segmentation metrics.
        """

        tp = self.tp
        tn = self.tn
        fp = self.fp
        fn = self.fn

        iou = (
            tp
            / (
                tp
                + fp
                + fn
                + 1e-8
            )
        )

        f1 = (
            2.0 * tp
            / (
                2.0 * tp
                + fp
                + fn
                + 1e-8
            )
        )

        precision = (
            tp
            / (
                tp
                + fp
                + 1e-8
            )
        )

        recall = (
            tp
            / (
                tp
                + fn
                + 1e-8
            )
        )

        accuracy = (
            (tp + tn)
            / (
                tp
                + tn
                + fp
                + fn
                + 1e-8
            )
        )

        return {
            "iou": float(iou),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "accuracy": float(accuracy),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }


# ============================================================
# Evaluator
# ============================================================


class SegmentationEvaluator:
    """
    Dataset-level evaluator for DINOv3 + LoRA forgery
    segmentation.

    The evaluator follows the Authenta evaluation pattern:

        YAML
          ↓
        model construction
          ↓
        checkpoint loading
          ↓
        test dataset
          ↓
        DataLoader
          ↓
        batch inference
          ↓
        segmentation metrics
          ↓
        evaluation artifacts

    Expected test batch:

        {
            "image": Tensor,
            "mask": Tensor,
        }

    Optional additional fields in the batch are ignored.

    The evaluator is intentionally dataset-level. It does not
    implement standalone single-image inference or sliding-window
    inference.
    """

    def __init__(
        self,
        yml_path: str,
        device: str = "cuda",
    ):

        self.yml_path = yml_path

        # --------------------------------------------------------
        # Configuration
        # --------------------------------------------------------

        self.opt = parse_yml(
            yml_path
        )

        self.device = (
            device
            if (
                device.startswith("cuda")
                and torch.cuda.is_available()
            )
            else "cpu"
        )

        eval_settings = (
            self.opt.get(
                "eval_settings"
            )
            or NoneDict()
        )

        self.image_size = int(
            eval_settings["image_size"]
            or 512
        )

        self.threshold = float(
            eval_settings["threshold"]
            if eval_settings["threshold"]
            is not None
            else 0.5
        )

        self.batch_size = int(
            eval_settings["batch_size"]
            or 4
        )

        self.num_workers = int(
            eval_settings["num_workers"]
            or 4
        )

        self.half_precision = bool(
            eval_settings["half_precision"]
            if eval_settings[
                "half_precision"
            ] is not None
            else True
        )

        # --------------------------------------------------------
        # Output directory
        # --------------------------------------------------------

        default_output_dir = (
            Path(
                self.opt[
                    "train_settings"
                ][
                    "save_checkpoint_folder_path"
                ]
                or "checkpoints"
            )
            / (
                self.opt["name"]
                or "experiment"
            )
            / "eval"
        )

        self.output_dir = Path(
            eval_settings["output_dir"]
            or default_output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------------
        # Logger
        # --------------------------------------------------------

        self.log = get_logger(
            name=(
                f"{self.opt['name'] or 'experiment'}"
                "_eval"
            ),
            log_dir=str(
                self.output_dir
            ),
        )

        # --------------------------------------------------------
        # Checkpoint
        # --------------------------------------------------------

        checkpoint_path = (
            eval_settings[
                "checkpoint_path"
            ]
        )

        if not checkpoint_path:
            raise ValueError(
                "eval_settings.checkpoint_path "
                f"is required in {yml_path}"
            )

        self.checkpoint_path = (
            checkpoint_path
        )

        self._log(
            "Checkpoint: "
            f"{self.checkpoint_path}"
        )

        # --------------------------------------------------------
        # Build model
        # --------------------------------------------------------

        self.model = (
            self._build_model()
        )

        # --------------------------------------------------------
        # Build test dataset
        # --------------------------------------------------------

        self.test_dataset = (
            self._build_test_dataset()
        )

        # --------------------------------------------------------
        # Build DataLoader
        # --------------------------------------------------------

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(
                self.device.startswith(
                    "cuda"
                )
            ),
            persistent_workers=(
                self.num_workers > 0
            ),
        )

        self._log(
            f"Test samples: "
            f"{len(self.test_dataset)}"
        )

        self._log(
            f"Batch size: "
            f"{self.batch_size} | "
            f"workers: "
            f"{self.num_workers}"
        )

        self._log(
            f"Image size: "
            f"{self.image_size} | "
            f"threshold: "
            f"{self.threshold}"
        )

        self._log(
            f"Device: "
            f"{self.device} | "
            f"half precision: "
            f"{self.half_precision}"
        )

    # ========================================================
    # Logging
    # ========================================================

    def _log(
        self,
        message: str,
    ) -> None:

        self.log.info(
            message
        )

    # ========================================================
    # Model construction
    # ========================================================

    def _build_model(self):

        structure = (
            self.opt.get(
                "structure"
            )
            or NoneDict()
        )

        backbone = (
            structure.get(
                "backbone"
            )
            or NoneDict()
        )

        model_path = backbone[
            "model_path"
        ]

        if not model_path:
            raise ValueError(
                "structure.backbone.model_path must point to the "
                "local Hugging Face DINOv3 ViT-L/16 model directory."
            )

        model = build_dinov3_segmentation(
            backbone_path=model_path,
            lora_rank=int(
                backbone[
                    "lora_rank"
                ]
                or 32
            ),
            lora_alpha=float(
                backbone[
                    "lora_alpha"
                ]
                or 64.0
            ),
            lora_dropout=float(
                backbone[
                    "lora_dropout"
                ]
                or 0.0
            ),
        )

        load_checkpoint(
            model,
            self.checkpoint_path,
            strict=True,
        )

        model = (
            model
            .to(self.device)
            .eval()
        )

        return model

    # ========================================================
    # Dataset construction
    # ========================================================

    def _build_test_dataset(self):

        dataset_config = (
            self.opt[
                "datasets"
            ]["test"]
        )

        if not dataset_config:
            raise ValueError(
                "datasets.test is required "
                f"in {self.yml_path}"
            )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # The exact dataset class will be wired here once
        # authgenforge/data/ is finalized.
        #
        # This keeps evaluator.py independent from the
        # underlying storage implementation.
        # ----------------------------------------------------

        dataset_type = (
            dataset_config[
                "type"
            ]
        )

        if not dataset_type:
            raise ValueError(
                "datasets.test.type is required "
                f"in {self.yml_path}"
            )

        # Dynamic dataset import follows the same general
        # pattern used by the Authenta video evaluator.

        module = importlib.import_module(
            f"authgenforge.data.{dataset_type}"
        )

        if not hasattr(
            module,
            "ForensicsDataset",
        ):
            raise AttributeError(
                f"authgenforge.data.{dataset_type} "
                "must expose ForensicsDataset"
            )

        dataset_class = (
            module.ForensicsDataset
        )

        parameters = (
            dataset_config[
                "parameters"
            ]
            or {}
        )

        return dataset_class(
            **parameters
        )

    # ========================================================
    # Preprocessing
    # ========================================================

    def _prepare_batch(
        self,
        batch,
    ):

        image = batch[
            "image"
        ]

        mask = batch[
            "mask"
        ]

        image = image.to(
            self.device,
            non_blocking=True,
        )

        mask = mask.to(
            self.device,
            non_blocking=True,
        )

        return image, mask

    # ========================================================
    # Model prediction
    # ========================================================

    @torch.inference_mode()
    def _predict(
        self,
        image: torch.Tensor,
    ):

        autocast_enabled = (
            self.half_precision
            and self.device.startswith(
                "cuda"
            )
        )

        with torch.autocast(
            device_type=(
                self.device.split(
                    ":",
                    1,
                )[0]
            ),
            dtype=torch.float16,
            enabled=autocast_enabled,
        ):

            logits = self.model(
                image
            )

        # Support models returning either:
        #
        #   logits
        #
        # or
        #
        #   (logits, ...)

        if isinstance(
            logits,
            (tuple, list),
        ):
            logits = logits[0]

        probabilities = torch.sigmoid(
            logits
        )

        return probabilities

    # ========================================================
    # Memory cleanup
    # ========================================================

    def _clear_memory(self):

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # Evaluation
    # ========================================================

    def run(self) -> dict:
        """
        Evaluate the complete test dataset.
        """

        metrics_accumulator = (
            _SegmentationMetricsAccumulator(
                threshold=self.threshold
            )
        )

        predictions_path = (
            self.output_dir
            / "predictions.csv"
        )

        metrics_path = (
            self.output_dir
            / "metrics.json"
        )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        with open(
            predictions_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "batch",
                    "batch_size",
                ],
            )

            writer.writeheader()
            csv_file.flush()

            # ------------------------------------------------
            # Evaluation
            # ------------------------------------------------

            with torch.inference_mode():

                pbar = tqdm(
                    self.test_loader,
                    desc="Evaluation",
                )

                for batch_idx, batch in enumerate(
                    pbar
                ):

                    try:

                        image, mask = (
                            self._prepare_batch(
                                batch
                            )
                        )

                        probabilities = (
                            self._predict(
                                image
                            )
                        )

                        metrics_accumulator.update(
                            probabilities,
                            mask,
                        )

                        writer.writerow(
                            {
                                "batch": batch_idx,
                                "batch_size": (
                                    image.shape[0]
                                ),
                            }
                        )

                        csv_file.flush()

                        current_metrics = (
                            metrics_accumulator.metrics()
                        )

                        pbar.set_postfix(
                            {
                                "IoU": (
                                    f"{current_metrics['iou']:.4f}"
                                ),
                                "F1": (
                                    f"{current_metrics['f1']:.4f}"
                                ),
                            }
                        )

                        del (
                            image,
                            mask,
                            probabilities,
                        )

                        self._clear_memory()

                    except Exception as exc:

                        self._log(
                            "WARNING: failed to "
                            f"evaluate batch "
                            f"{batch_idx}: "
                            f"{exc!r}"
                        )

                        self._clear_memory()

                        continue

        # ----------------------------------------------------
        # Final metrics
        # ----------------------------------------------------

        metrics = (
            metrics_accumulator.metrics()
        )

        metrics["num_samples"] = len(
            self.test_dataset
        )

        metrics["threshold"] = (
            self.threshold
        )

        metrics["image_size"] = (
            self.image_size
        )

        # ----------------------------------------------------
        # Save metrics
        # ----------------------------------------------------

        with open(
            metrics_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metrics,
                file,
                indent=2,
            )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        experiment_name = (
            self.opt["name"]
            or "experiment"
        )

        print(
            "\n"
            f"=== {experiment_name} "
            "segmentation evaluation ===\n"
            f"samples   : "
            f"{metrics['num_samples']}\n"
            f"IoU       : "
            f"{metrics['iou']:.4f}\n"
            f"F1        : "
            f"{metrics['f1']:.4f}\n"
            f"Precision : "
            f"{metrics['precision']:.4f}\n"
            f"Recall    : "
            f"{metrics['recall']:.4f}\n"
            f"Accuracy  : "
            f"{metrics['accuracy']:.4f}\n"
        )

        self._log(
            f"IoU {metrics['iou']:.4f} | "
            f"F1 {metrics['f1']:.4f} | "
            f"Precision "
            f"{metrics['precision']:.4f} | "
            f"Recall "
            f"{metrics['recall']:.4f} | "
            f"Accuracy "
            f"{metrics['accuracy']:.4f}"
        )

        return metrics


# ============================================================
# Public entry point
# ============================================================


def evaluate_from_yml(
    yml_path: str,
    device: str = "cuda",
) -> dict:

    return SegmentationEvaluator(
        yml_path,
        device=device,
    ).run()