
from __future__ import annotations

import logging

from authgenforge import *

from authgenforge.data.dataloader import (
    build_dataloaders,
)

from authgenforge.losses import (
    ForgerySegmentationLoss,
)

from authgenforge.networks.dinov3_segmentation import (
    build_dinov3_segmentation,
    load_checkpoint,
)

from authgenforge.optimizers.optimizer import (
    get_optimizer,
)

from authgenforge.optimizers.scheduler import (
    CosineLRScheduler,
)

from authgenforge.options.option_utils import (
    parse_yml,
)

from authgenforge.training import (
    SegmentationTrainer,
)


log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------------


def _cfg(value, default):
    """
    Fall back to `default` only when `value` is unset (None).

    Unlike `value or default`, this preserves explicit falsy values
    such as 0 and False.
    """

    return (
        default
        if value is None
        else value
    )


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------


def get_dataloaders_from_yml(
    yml_path: str,
):
    """
    Build train and test DataLoaders from the YAML configuration.

    The actual dataset implementation remains inside
    authgenforge.data and is intentionally kept separate from
    the model/training configuration logic.
    """

    if yml_path is None:
        raise ValueError(
            "yml_path is required"
        )

    opt = parse_yml(
        yml_path
    )

    train_cfg = opt["datasets"]["train"]
    test_cfg = opt["datasets"]["test"]

    data_context = _cfg(
        opt.get("data_context"),
        "normal",
    )

    train_loader, test_loader = build_dataloaders(
        train_dir=train_cfg["dataroot"],
        test_dir=test_cfg["dataroot"],

        crop_size=_cfg(
            train_cfg.get("crop_size"),
            512,
        ),

        batch_size=_cfg(
            train_cfg.get("batch_size"),
            4,
        ),

        num_workers=_cfg(
            train_cfg.get("n_workers"),
            4,
        ),

        test_num_workers=_cfg(
            test_cfg.get("n_workers"),
            None,
        ),

        pin_memory=_cfg(
            train_cfg.get("pin_memory"),
            True,
        ),

        buffer_size=_cfg(
            train_cfg.get("buffer_size"),
            1000,
        ),

        stateful=_cfg(
            train_cfg.get("stateful_loader"),
            True,
        ),

        data_context=data_context,
    )

    return (
        train_loader,
        test_loader,
    )


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------


def get_model_from_yml(
    yml_path: str,
):
    """
    Build the DINOv3 ViT-L/16 + LoRA segmentation model from YAML.

    The backbone path points to a local Hugging Face model directory
    containing:

        config.json
        model.safetensors

    The actual pretrained model artifact can be supplied externally,
    for example through S3 in SageMaker.
    """

    if yml_path is None:
        raise ValueError(
            "yml_path is required"
        )

    opt = parse_yml(
        yml_path
    )

    backbone = opt["structure"]["backbone"]

    pretrain = opt["pretraining_settings"]

    model_path = backbone.get("model_path")

    if not model_path:
        raise ValueError(
            "structure.backbone.model_path must point to the "
            "local Hugging Face DINOv3 ViT-L/16 model directory."
        )

    model = build_dinov3_segmentation(
        backbone_path=model_path,

        lora_rank=int(
            _cfg(
                backbone.get("lora_rank"),
                32,
            )
        ),

        lora_alpha=float(
            _cfg(
                backbone.get("lora_alpha"),
                64.0,
            )
        ),

        lora_dropout=float(
            _cfg(
                backbone.get("lora_dropout"),
                0.0,
            )
        ),
    )

    # Optional Authenta checkpoint loading.
    #
    # This is different from the base DINOv3 model weights.
    # model_path provides the pretrained DINOv3 backbone,
    # while checkpoint_path can provide a previously trained
    # Authenta segmentation checkpoint.
    if (
        _cfg(
            pretrain.get("want_load"),
            False,
        )
        and pretrain.get("checkpoint_path")
    ):
        load_checkpoint(
            model,
            pretrain["checkpoint_path"],
            strict=_cfg(
                pretrain.get("strict_load"),
                True,
            ),
        )

    return model


# ------------------------------------------------------------------
# Criterion
# ------------------------------------------------------------------


def get_criterion_from_yml(
    yml_path: str,
) -> nn.Module:
    """
    Build the forgery segmentation criterion from YAML.
    """

    if yml_path is None:
        raise ValueError(
            "yml_path is required"
        )

    opt = parse_yml(
        yml_path
    )

    train_cfg = opt["train_settings"]

    criterion_name = (
        _cfg(
            train_cfg.get("criterion"),
            "forgery_segmentation",
        )
        .lower()
    )

    if criterion_name not in {
        "forgery_segmentation",
        "forgery",
        "bce_edge",
    }:
        raise ValueError(
            "Unknown criterion "
            f"'{criterion_name}'. "
            "For this project use "
            "'forgery_segmentation'."
        )

    loss_cfg = (
        train_cfg.get(
            "forgery_segmentation_loss"
        )
        or {}
    )

    edge_lambda = float(
        _cfg(
            loss_cfg.get("edge_lambda"),
            20.0,
        )
    )

    return ForgerySegmentationLoss(
        edge_lambda=edge_lambda
    )


# ------------------------------------------------------------------
# Steps per epoch
# ------------------------------------------------------------------


_DEFAULT_STEPS_PER_EPOCH = 10_000


def get_steps_per_epoch_from_yml(
    yml_path: str,
) -> int:
    """
    Fallback for streaming datasets that do not implement __len__.

    The returned value represents optimizer-update steps per epoch,
    not raw DataLoader batches, because the scheduler advances once
    per optimizer update.
    """

    if yml_path is None:
        raise ValueError(
            "yml_path is required"
        )

    opt = parse_yml(
        yml_path
    )

    train_cfg = opt["datasets"]["train"]
    train_settings = opt["train_settings"]

    total_samples = train_cfg.get(
        "total_samples"
    )

    batch_size = int(
        _cfg(
            train_cfg.get("batch_size"),
            4,
        )
    )

    grad_accum_steps = int(
        _cfg(
            train_settings.get("grad_accum_steps"),
            1,
        )
    )

    if total_samples is None:
        log.warning(
            "datasets.train.total_samples not set in %s "
            "and steps_per_epoch not given; falling back "
            "to steps_per_epoch=%d. The LR schedule will "
            "not match the actual dataset size — set "
            "total_samples for correct results.",
            yml_path,
            _DEFAULT_STEPS_PER_EPOCH,
        )

        return _DEFAULT_STEPS_PER_EPOCH

    # Number of DataLoader batches.
    batches_per_epoch = max(
        (
            int(total_samples)
            + batch_size
            - 1
        )
        // batch_size,
        1,
    )

    # Number of actual optimizer updates.
    optimizer_steps = max(
        (
            batches_per_epoch
            + grad_accum_steps
            - 1
        )
        // grad_accum_steps,
        1,
    )

    return optimizer_steps


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------


def get_trainer_from_yml(
    yml_path: str,
    model,
    train_loader,
    test_loader,
    steps_per_epoch: int,
) -> SegmentationTrainer:
    """
    Construct the complete segmentation trainer.
    """

    if yml_path is None:
        raise ValueError(
            "yml_path is required"
        )

    opt = parse_yml(
        yml_path
    )

    ts = opt["train_settings"]
    es = opt["epoch_settings"]
    ds = opt["datasets"]["train"]

    # --------------------------------------------------------------
    # Criterion
    # --------------------------------------------------------------

    criterion = get_criterion_from_yml(
        yml_path
    )

    # --------------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------------

    base_lr = float(
        _cfg(
            ts.get("base_lr"),
            5e-4,
        )
    )

    optimizer = get_optimizer(
        model=model,
        base_lr=base_lr,

        weight_decay=float(
            _cfg(
                ts.get("weight_decay"),
                0.05,
            )
        ),

        layer_decay=float(
            _cfg(
                ts.get("layer_decay"),
                0.8,
            )
        ),
    )

    # --------------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------------

    scheduler = CosineLRScheduler(
        optimizer=optimizer,

        num_epochs=int(
            _cfg(
                es.get("total_epochs"),
                50,
            )
        ),

        steps_per_epoch=steps_per_epoch,

        base_lr=base_lr,

        min_lr=float(
            _cfg(
                ts.get("min_lr"),
                2.5e-7,
            )
        ),

        warmup_epochs=float(
            _cfg(
                ts.get("warmup_epochs"),
                0.001,
            )
        ),

        warmup_lr=float(
            _cfg(
                ts.get("warmup_lr"),
                2.5e-7,
            )
        ),

        scheduler_type=_cfg(
            ts.get("scheduler_type"),
            "cyclic",
        ),

        T_0_epochs=int(
            _cfg(
                ts.get("T_0_epochs"),
                1,
            )
        ),
    )

    # --------------------------------------------------------------
    # Resume
    # --------------------------------------------------------------

    resume = ts.get(
        "load_checkpoint_file_path"
    )

    # --------------------------------------------------------------
    # Trainer
    # --------------------------------------------------------------

    return SegmentationTrainer(
        model=model,

        train_loader=train_loader,

        test_loader=test_loader,

        criterion=criterion,

        optimizer=optimizer,

        scheduler=scheduler,

        experiment_name=_cfg(
            opt.get("name"),
            "experiment",
        ),

        save_checkpoint_dir=_cfg(
            ts.get(
                "save_checkpoint_folder_path"
            ),
            "checkpoints",
        ),

        mixed_precision=_cfg(
            ts.get("mixed_precision"),
            True,
        ),

        grad_accum_steps=int(
            _cfg(
                ts.get("grad_accum_steps"),
                1,
            )
        ),

        grad_clip=float(
            _cfg(
                ts.get("grad_clip"),
                1.0,
            )
        ),

        log_interval=int(
            _cfg(
                ts.get("log_interval"),
                100,
            )
        ),

        save_interval=int(
            _cfg(
                ts.get("save_interval"),
                2000,
            )
        ),

        load_checkpoint_path=(
            resume
            if resume
            else None
        ),
    )


# ------------------------------------------------------------------
# Quick sample
# ------------------------------------------------------------------


def get_sample_from_yml(
    yml_path: str,
) -> dict:
    """
    Load a single training batch for a quick pipeline check.

    The underlying dataset implementation is intentionally kept
    outside this file.
    """

    if yml_path is None:
        raise ValueError(
            "yml_path is required"
        )

    opt = parse_yml(
        yml_path
    )

    train_cfg = opt["datasets"]["train"]
    test_cfg = opt["datasets"]["test"]

    from authgenforge.data.forensics_dataset import (
        build_forensics_datasets,
    )

    train_ds, _ = build_forensics_datasets(
        train_dir=train_cfg["dataroot"],

        test_dir=test_cfg["dataroot"],

        crop_size=_cfg(
            train_cfg.get("crop_size"),
            512,
        ),

        buffer_size=_cfg(
            train_cfg.get("buffer_size"),
            1000,
        ),

        data_context=_cfg(
            opt.get("data_context"),
            "normal",
        ),
    )

    loader = DataLoader(
        train_ds,

        batch_size=_cfg(
            train_cfg.get("batch_size"),
            4,
        ),

        num_workers=0,
    )

    return next(
        iter(loader)
    )


# ------------------------------------------------------------------
# Full pipeline
# ------------------------------------------------------------------


def load_pipeline_from_yml(
    yml_path: str,
    steps_per_epoch: int | None = None,
):
    """
    Build the complete training pipeline:

        DataLoaders
            ↓
        DINOv3 ViT-L/16 + LoRA
            ↓
        Criterion
            ↓
        Optimizer
            ↓
        Scheduler
            ↓
        SegmentationTrainer
    """

    if yml_path is None:
        raise ValueError(
            "yml_path is required"
        )

    train_loader, test_loader = (
        get_dataloaders_from_yml(
            yml_path
        )
    )

    # --------------------------------------------------------------
    # Determine optimizer-update steps per epoch.
    # --------------------------------------------------------------

    if steps_per_epoch is None:
        try:
            num_batches = len(
                train_loader
            )

            opt = parse_yml(
                yml_path
            )

            grad_accum_steps = int(
                _cfg(
                    opt["train_settings"].get(
                        "grad_accum_steps"
                    ),
                    1,
                )
            )

            # Scheduler steps once per optimizer update.
            steps_per_epoch = max(
                (
                    num_batches
                    + grad_accum_steps
                    - 1
                )
                // grad_accum_steps,
                1,
            )

        except TypeError:
            # Streaming dataset — no __len__.
            steps_per_epoch = (
                get_steps_per_epoch_from_yml(
                    yml_path
                )
            )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    model = get_model_from_yml(
        yml_path
    )

    # --------------------------------------------------------------
    # Trainer
    # --------------------------------------------------------------

    trainer = get_trainer_from_yml(
        yml_path=yml_path,

        model=model,

        train_loader=train_loader,

        test_loader=test_loader,

        steps_per_epoch=steps_per_epoch,
    )

    return (
        train_loader,
        test_loader,
        model,
        trainer,
    )