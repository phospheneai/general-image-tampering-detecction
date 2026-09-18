#!/usr/bin/env python
"""
SageMaker training-job entry point for the Authenta general image
forgery segmentation model.

The SageMaker estimator uses:

    source_dir = sagemaker/

Therefore this file runs from /opt/ml/code/ inside the SageMaker
training container.

The authgenforge package is supplied separately through the estimator's
dependencies and is available under /opt/ml/code/authgenforge/.

This script:

1. Loads the SageMaker-side training configuration.
2. Handles Spot-training checkpoint resume.
3. Disables optional initialization weights when they are unavailable.
4. Builds the segmentation training pipeline through authgenforge.
5. Runs SegmentationTrainer.train_model().
6. Writes tracebacks to /opt/ml/output/failure when training fails.

Nothing under the root configs/ directory or DATASETS.md is modified.
"""


from __future__ import annotations


# ============================================================
# Standard library
# ============================================================

import argparse
import os
import sys
import tempfile
import traceback

from pathlib import Path


# ============================================================
# Third-party
# ============================================================

import yaml


# ============================================================
# SageMaker source directory
# ============================================================

# On SageMaker:
#     /opt/ml/code/
#
# Locally:
#     <repo>/sagemaker/
#
# source_dir=sagemaker/ means the SageMaker container receives
# the contents of this directory under /opt/ml/code/.

CODE_DIR = Path(__file__).resolve().parent


# ============================================================
# YAML helpers
# ============================================================

def _load_yaml(path: Path) -> dict:
    """
    Load a YAML configuration file.
    """
    data = yaml.safe_load(
        path.read_text(
            encoding="utf-8",
        )
    )

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected YAML mapping in {path}"
        )

    return data


def _effective_config(cfg_path: Path) -> Path:
    """
    Build the effective SageMaker training configuration.

    The committed SageMaker config is kept unchanged unless an
    environment-specific adjustment is required.

    Adjustments currently supported:

    1. Spot restart:
       If SageMaker has restored a latest checkpoint under the
       configured checkpoint directory, wire that checkpoint into
       the training configuration and enable dataloader resume.

    2. Optional initialization weights:
       If want_load=True but the configured initialization checkpoint
       is not available, disable want_load rather than failing before
       the training pipeline is constructed.

    A temporary YAML file is created only when a patch is required.
    """

    opt = _load_yaml(
        cfg_path,
    )

    name = opt.get(
        "name"
    ) or "experiment"

    train_settings = (
        opt.get("train_settings")
        or {}
    )

    checkpoint_root = train_settings.get(
        "save_checkpoint_folder_path"
    )

    patched = False

    # --------------------------------------------------------
    # Spot restart
    # --------------------------------------------------------

    latest = (
        Path(checkpoint_root)
        / name
        / "latest_checkpoint.pth"
        if checkpoint_root
        else None
    )

    if latest and latest.is_file():
        print(
            f"[train] checkpoint resume detected: {latest}",
            flush=True,
        )

        train_settings[
            "load_checkpoint_file_path"
        ] = str(latest)

        train_settings[
            "resume_dataloader"
        ] = True

        opt[
            "train_settings"
        ] = train_settings

        patched = True

    # --------------------------------------------------------
    # Optional initialization weights
    # --------------------------------------------------------

    pretraining_settings = (
        opt.get("pretraining_settings")
        or {}
    )

    checkpoint_path = pretraining_settings.get(
        "checkpoint_path"
    )

    want_load = pretraining_settings.get(
        "want_load",
        False,
    )

    if (
        want_load
        and checkpoint_path
        and not Path(checkpoint_path).is_file()
    ):
        print(
            "[train] initialization checkpoint "
            f"{checkpoint_path} not present; "
            "setting want_load=false",
            flush=True,
        )

        pretraining_settings[
            "want_load"
        ] = False

        opt[
            "pretraining_settings"
        ] = pretraining_settings

        patched = True

    # --------------------------------------------------------
    # No patch required
    # --------------------------------------------------------

    if not patched:
        return cfg_path

    # --------------------------------------------------------
    # Write temporary effective configuration
    # --------------------------------------------------------

    fd, tmp_path = tempfile.mkstemp(
        prefix="effective_config_",
        suffix=".yml",
    )

    with os.fdopen(
        fd,
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            opt,
            file,
            sort_keys=False,
        )

    return Path(tmp_path)


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    SageMaker training entry point.
    """

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config/normal/train_forensics.yml",
        help=(
            "Training YAML configuration relative to "
            "the SageMaker source directory."
        ),
    )

    parser.add_argument(
        "--end-epoch",
        type=int,
        default=None,
        help=(
            "Stop after this epoch. "
            "Defaults to epoch_settings.total_epochs."
        ),
    )

    # SageMaker injects additional arguments such as --model_dir.
    # parse_known_args() prevents those arguments from breaking
    # this training entry point.
    args, _ = parser.parse_known_args()

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    os.environ.setdefault(
        "HF_HOME",
        "/tmp/hf",
    )

    os.environ.setdefault(
        "HF_DATASETS_CACHE",
        "/tmp/hf/datasets",
    )

    os.environ.setdefault(
        "TQDM_MININTERVAL",
        "60",
    )

    # --------------------------------------------------------
    # Python paths
    # --------------------------------------------------------

    # /opt/ml/code contains:
    #
    #   train.py
    #   config/
    #   backbone/
    #
    # authgenforge is supplied through the estimator's
    # dependencies and is also expected under /opt/ml/code/.
    #
    # The parent is added as well for local execution from the
    # repository's sagemaker directory.

    for path in (
        CODE_DIR,
        CODE_DIR.parent,
    ):
        path_string = str(path)

        if path_string not in sys.path:
            sys.path.insert(
                0,
                path_string,
            )

    # --------------------------------------------------------
    # Configuration path
    # --------------------------------------------------------

    cfg_path = (
        CODE_DIR / args.config
    ).resolve()

    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"config not found: {cfg_path}"
        )

    # --------------------------------------------------------
    # End epoch
    # --------------------------------------------------------

    end_epoch = args.end_epoch

    if end_epoch is None:
        config = _load_yaml(
            cfg_path,
        )

        epoch_settings = (
            config.get("epoch_settings")
            or {}
        )

        end_epoch = int(
            epoch_settings.get(
                "total_epochs"
            )
            or 1
        )

    # --------------------------------------------------------
    # Effective configuration
    # --------------------------------------------------------

    effective_config = _effective_config(
        cfg_path,
    )

    print(
        f"[train] config={effective_config}",
        flush=True,
    )

    print(
        f"[train] end_epoch={end_epoch}",
        flush=True,
    )

    # --------------------------------------------------------
    # Training pipeline
    # --------------------------------------------------------

    try:
        from authgenforge.options.load import (
            load_pipeline_from_yml,
        )

        (
            _,
            _,
            _,
            trainer,
        ) = load_pipeline_from_yml(
            str(effective_config)
        )

        print(
            "[train] "
            f"experiment={trainer.experiment_name} "
            f"ckpt_dir={trainer.ckpt_dir}",
            flush=True,
        )

        trainer.train_model(
            end_epoch=end_epoch,
        )

    except Exception:
        traceback_text = traceback.format_exc()

        print(
            traceback_text,
            flush=True,
        )

        failure_path = Path(
            "/opt/ml/output/failure"
        )

        if failure_path.parent.is_dir():
            failure_path.write_text(
                traceback_text,
                encoding="utf-8",
            )

        raise

    finally:
        # Remove the temporary patched configuration if one
        # was created.
        if effective_config != cfg_path:
            effective_config.unlink(
                missing_ok=True,
            )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()