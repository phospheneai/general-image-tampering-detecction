#!/usr/bin/env python
"""
Launch the Authenta General Image Forgery training job on SageMaker.

Example:

    python sagemaker/launch.py \
        --role arn:aws:iam::<ACCT>:role/<ExecutionRole> \
        --run-name <name>

Region, bucket, and input channels come from the `sagemaker:` block of

    sagemaker/config/normal/train_forensics.yml

The configuration path is relative to `sagemaker/`, which is also the
estimator's source_dir.

The training code uses:

    source_dir = sagemaker/

while the `authgenforge/` package is supplied through the estimator's
dependencies and becomes available inside the SageMaker container under:

    /opt/ml/code/authgenforge/

The run name is used as:

1. The SageMaker training job name.
2. The S3 checkpoint sub-prefix.
3. The S3 output sub-prefix.

This keeps different runs isolated and allows a Spot restart of the
same training job to restore its own checkpoints.

The trainer is single-process, so instance_count is fixed at 1.

The pretrained DINOv3 ViT-L/16 Hugging Face model is supplied through
the SageMaker `artifacts` input channel. The channel is mounted inside
the training container at:

    /opt/ml/input/data/artifacts/

The SageMaker-side training configuration points the model to that
local directory through:

    structure.backbone.model_path
"""


from __future__ import annotations


# ============================================================
# Standard library
# ============================================================

import argparse
import os
import re
import sys

from pathlib import Path


# ============================================================
# Third-party
# ============================================================

import yaml


# ============================================================
# Repository paths
# ============================================================

REPO_ROOT = Path(
    __file__
).resolve().parent.parent

SM_SRC = REPO_ROOT / "sagemaker"

DEFAULT_CONFIG = (
    "config/normal/train_forensics.yml"
)


# ============================================================
# SageMaker framework configuration
# ============================================================

FRAMEWORK_VERSION = "2.3"
PY_VERSION = "py311"


# ============================================================
# Metrics
# ============================================================

# These regexes correspond to the segmentation trainer's
# per-epoch logging output.
#
# The exact trainer log format should be kept aligned with
# these definitions when logging is finalized.

METRIC_DEFS = [
    {
        "Name": "train:loss",
        "Regex": (
            r"train.*loss[:=]\s*([0-9eE+.\-]+)"
        ),
    },
    {
        "Name": "train:iou",
        "Regex": (
            r"train.*iou[:=]\s*([0-9eE+.\-]+)"
        ),
    },
    {
        "Name": "train:f1",
        "Regex": (
            r"train.*f1[:=]\s*([0-9eE+.\-]+)"
        ),
    },
    {
        "Name": "val:loss",
        "Regex": (
            r"val.*loss[:=]\s*([0-9eE+.\-]+)"
        ),
    },
    {
        "Name": "val:iou",
        "Regex": (
            r"val.*iou[:=]\s*([0-9eE+.\-]+)"
        ),
    },
    {
        "Name": "val:f1",
        "Regex": (
            r"val.*f1[:=]\s*([0-9eE+.\-]+)"
        ),
    },
    {
        "Name": "val:precision",
        "Regex": (
            r"val.*precision[:=]\s*([0-9eE+.\-]+)"
        ),
    },
    {
        "Name": "val:recall",
        "Regex": (
            r"val.*recall[:=]\s*([0-9eE+.\-]+)"
        ),
    },
    {
        "Name": "val:accuracy",
        "Regex": (
            r"val.*accuracy[:=]\s*([0-9eE+.\-]+)"
        ),
    },
]


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=(
            "SageMaker training YAML, relative to "
            "sagemaker/."
        ),
    )

    parser.add_argument(
        "--role",
        default=os.environ.get(
            "SAGEMAKER_ROLE"
        ),
        help=(
            "SageMaker execution role ARN "
            "(env: SAGEMAKER_ROLE)."
        ),
    )

    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Job name and S3 sub-prefix for this run. "
            "Prompted if omitted or already taken."
        ),
    )

    parser.add_argument(
        "--end-epoch",
        type=int,
        default=None,
        help=(
            "Stop after this epoch. "
            "Defaults to epoch_settings.total_epochs "
            "from the SageMaker config."
        ),
    )

    parser.add_argument(
        "--instance-type",
        default="ml.g6e.2xlarge",
        help=(
            "Single-GPU SageMaker instance type."
        ),
    )

    parser.add_argument(
        "--volume-size",
        type=int,
        default=100,
        help="EBS volume size in GB.",
    )

    parser.add_argument(
        "--max-run-hours",
        type=float,
        default=48,
        help="Maximum training runtime in hours.",
    )

    parser.add_argument(
        "--max-wait-hours",
        type=float,
        default=72,
        help=(
            "Maximum Spot wait time in hours; "
            "must be >= max-run-hours."
        ),
    )

    parser.add_argument(
        "--no-spot",
        action="store_true",
        help=(
            "Use an on-demand instance instead of "
            "Managed Spot Training."
        ),
    )

    parser.add_argument(
        "--no-wait",
        action="store_true",
        help=(
            "Submit the training job and return "
            "without streaming training logs."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the planned SageMaker configuration "
            "and exit without submitting a job."
        ),
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def _u(uri: str) -> str:
    """
    Ensure an S3 URI ends with a slash.
    """
    return uri if uri.endswith("/") else uri + "/"


def _sanitize(name: str) -> str:
    """
    Convert a proposed run name into a SageMaker-compatible
    training job name.
    """

    sanitized = re.sub(
        r"[^a-zA-Z0-9-]+",
        "-",
        name,
    )

    sanitized = sanitized.strip("-")

    return sanitized[:63]


def _resolve_run_name(
    run_name: str | None,
    region: str,
) -> str:
    """
    Resolve a unique SageMaker training job name.

    If the requested name is already in use, prompt for another
    name. When no interactive terminal is available, fail with
    a clear instruction to provide --run-name.
    """

    import boto3
    from botocore.exceptions import ClientError

    sm = boto3.Session(
        region_name=region,
    ).client(
        "sagemaker",
    )

    def taken(name: str):
        try:
            return sm.describe_training_job(
                TrainingJobName=name,
            )

        except ClientError as exc:
            error_code = (
                exc.response
                .get("Error", {})
                .get("Code")
            )

            # SageMaker uses ValidationException when the
            # requested training job does not exist.
            if error_code == "ValidationException":
                return None

            raise

    def ask(reason: str):
        if not sys.stdin.isatty():
            raise SystemExit(
                f"[launch] {reason} "
                "pass --run-name <name>."
            )

        print(
            f"[launch] {reason}"
        )

        answer = input(
            "[launch] run name: "
        ).strip()

        if not answer:
            raise SystemExit(
                "[launch] aborted."
            )

        answer = _sanitize(
            answer
        )

        if not answer:
            raise SystemExit(
                "[launch] invalid run name."
            )

        return answer

    if not run_name:
        run_name = ask(
            "name this training run —"
        )

    while (existing := taken(run_name)) is not None:
        created = existing.get(
            "CreationTime"
        )

        created_text = (
            created.strftime(
                "%Y-%m-%d %H:%M"
            )
            if created
            else "unknown"
        )

        status = existing.get(
            "TrainingJobStatus",
            "unknown",
        )

        run_name = ask(
            f"'{run_name}' is already a job "
            f"(status {status}, created {created_text}); "
            "names can't be reused —"
        )

    return run_name


def _load_config(path: Path) -> dict:
    """
    Load the SageMaker YAML configuration.
    """

    if not path.is_file():
        raise SystemExit(
            f"[launch] config not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise SystemExit(
            f"[launch] invalid YAML configuration: {path}"
        )

    return config


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config_path = (
        SM_SRC / args.config
    ).resolve()

    opt = _load_config(
        config_path
    )

    sm = (
        opt.get("sagemaker")
        or {}
    )

    region = sm.get(
        "region"
    )

    bucket = sm.get(
        "output_bucket"
    )

    checkpoint_prefix = (
        sm.get(
            "checkpoint_prefix",
            "general-forensics/checkpoints",
        )
        or "general-forensics/checkpoints"
    ).strip("/")

    output_prefix = (
        sm.get(
            "output_prefix",
            "general-forensics/output",
        )
        or "general-forensics/output"
    ).strip("/")

    inputs = (
        sm.get("inputs")
        or {}
    )

    # --------------------------------------------------------
    # Configuration validation
    # --------------------------------------------------------

    if not region:
        raise SystemExit(
            f"{args.config} needs sagemaker.region"
        )

    if not bucket:
        raise SystemExit(
            f"{args.config} needs "
            "sagemaker.output_bucket"
        )

    if not inputs:
        raise SystemExit(
            f"{args.config} needs "
            "sagemaker.inputs"
        )

    if "artifacts" not in inputs:
        raise SystemExit(
            f"{args.config} needs "
            "sagemaker.inputs.artifacts "
            "for the DINOv3 ViT-L/16 Hugging Face model."
        )

    artifacts_spec = (
        inputs.get("artifacts")
        or {}
    )

    if not artifacts_spec.get("uri"):
        raise SystemExit(
            f"{args.config} needs a non-empty "
            "sagemaker.inputs.artifacts.uri "
            "for the DINOv3 ViT-L/16 Hugging Face model."
        )

    if not args.role and not args.dry_run:
        raise SystemExit(
            "--role is required "
            "(or set SAGEMAKER_ROLE)"
        )

    # --------------------------------------------------------
    # Validate required project packaging
    # --------------------------------------------------------

    if not SM_SRC.is_dir():
        raise SystemExit(
            f"[launch] SageMaker source directory "
            f"not found: {SM_SRC}"
        )

    train_py = (
        SM_SRC / "train.py"
    )

    if not train_py.is_file():
        raise SystemExit(
            f"[launch] train.py not found: {train_py}"
        )

    authgenforge_dir = (
        REPO_ROOT / "authgenforge"
    )

    if not authgenforge_dir.is_dir():
        raise SystemExit(
            f"[launch] authgenforge package "
            f"not found: {authgenforge_dir}"
        )

    # --------------------------------------------------------
    # Run name
    # --------------------------------------------------------

    run_name = (
        _sanitize(args.run_name)
        if args.run_name
        else None
    )

    if args.dry_run:
        run_name = (
            run_name
            or "RUN-NAME"
        )
    else:
        run_name = _resolve_run_name(
            run_name,
            region,
        )

    if not run_name:
        raise SystemExit(
            "[launch] invalid run name."
        )

    # --------------------------------------------------------
    # Input channels
    # --------------------------------------------------------

    channels = {}

    for name, spec in inputs.items():
        if not spec:
            continue

        uri = spec.get(
            "uri"
        )

        if not uri:
            # Empty S3 URIs are intentionally rejected rather
            # than silently creating an incomplete training job.
            raise SystemExit(
                f"[launch] input channel '{name}' "
                "has an empty URI."
            )

        mode = spec.get(
            "mode",
            "File",
        )

        channels[name] = (
            _u(uri),
            mode,
        )

    if not channels:
        raise SystemExit(
            "[launch] no populated SageMaker input channels."
        )

    # --------------------------------------------------------
    # S3 paths
    # --------------------------------------------------------

    s3 = (
        f"s3://{bucket}"
    )

    checkpoint_s3_uri = (
        f"{s3}/"
        f"{checkpoint_prefix}/"
        f"{run_name}/"
    )

    output_s3_uri = (
        f"{s3}/"
        f"{output_prefix}/"
        f"{run_name}/"
    )

    # --------------------------------------------------------
    # End epoch
    # --------------------------------------------------------

    end_epoch = args.end_epoch

    if end_epoch is None:
        epoch_settings = (
            opt.get("epoch_settings")
            or {}
        )

        end_epoch = int(
            epoch_settings.get(
                "total_epochs"
            )
            or 1
        )

    # --------------------------------------------------------
    # SageMaker estimator configuration
    # --------------------------------------------------------

    estimator_kwargs = {
        "entry_point": "train.py",

        # The contents of sagemaker/ become /opt/ml/code/.
        "source_dir": str(SM_SRC),

        # authgenforge/ is outside source_dir, so ship it as
        # an estimator dependency.
        "dependencies": [
            str(authgenforge_dir),
        ],

        "role": args.role,

        "framework_version": FRAMEWORK_VERSION,
        "py_version": PY_VERSION,

        "instance_type": args.instance_type,
        "instance_count": 1,

        "volume_size": args.volume_size,

        "max_run": int(
            args.max_run_hours * 3600
        ),

        # SageMaker restores this checkpoint directory when a
        # Spot job is restarted.
        "checkpoint_s3_uri": checkpoint_s3_uri,
        "checkpoint_local_path": "/opt/ml/checkpoints",

        "output_path": output_s3_uri,

        # Keep generated training code in the output bucket.
        "code_location": (
            f"{s3}/"
            f"{output_prefix.split('/')[0]}/"
            "code"
        ),

        "disable_profiler": True,
        "debugger_hook_config": False,

        "metric_definitions": METRIC_DEFS,

        "hyperparameters": {
            "config": args.config,
            "end-epoch": end_epoch,
        },

        "environment": {
            "HF_HOME": "/tmp/hf",
            "HF_DATASETS_CACHE": "/tmp/hf/datasets",
            "TQDM_MININTERVAL": "60",
        },
    }

    # --------------------------------------------------------
    # Spot configuration
    # --------------------------------------------------------

    if not args.no_spot:
        if args.max_wait_hours < args.max_run_hours:
            raise SystemExit(
                "--max-wait-hours must be >= "
                "--max-run-hours"
            )

        estimator_kwargs.update(
            use_spot_instances=True,
            max_wait=int(
                args.max_wait_hours * 3600
            ),
        )

    # --------------------------------------------------------
    # Print launch configuration
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "[launch] Authenta General Image Forgery "
        "SageMaker Training"
    )
    print("=" * 72)

    print(
        f"[launch] config:          {config_path}"
    )

    print(
        f"[launch] region:          {region}"
    )

    print(
        f"[launch] run name:        {run_name}"
    )

    print(
        f"[launch] instance:        {args.instance_type}"
    )

    print(
        "[launch] instance count:  1"
    )

    print(
        f"[launch] framework:       PyTorch {FRAMEWORK_VERSION}"
    )

    print(
        f"[launch] Python:          {PY_VERSION}"
    )

    print(
        f"[launch] source dir:      {SM_SRC}"
    )

    print(
        f"[launch] dependency:      {authgenforge_dir}"
    )

    print(
        "[launch] backbone:        "
        "DINOv3 ViT-L/16 Hugging Face artifact"
    )

    print(
        "[launch] model mount:     "
        "/opt/ml/input/data/artifacts"
    )

    print(
        f"[launch] end epoch:       {end_epoch}"
    )

    print(
        f"[launch] spot:            {not args.no_spot}"
    )

    print(
        f"[launch] checkpoint S3:   {checkpoint_s3_uri}"
    )

    print(
        f"[launch] output S3:       {output_s3_uri}"
    )

    print()

    print("[launch] input channels:")

    for name, (
        uri,
        mode,
    ) in channels.items():
        print(
            f"    {name:12s} {uri} ({mode})"
        )

    print("=" * 72)
    print()

    # --------------------------------------------------------
    # Dry run
    # --------------------------------------------------------

    if args.dry_run:
        print(
            "[launch] dry run complete — "
            "no SageMaker job submitted."
        )
        return

    # --------------------------------------------------------
    # Import SageMaker only when submitting
    # --------------------------------------------------------

    import boto3
    import sagemaker

    from sagemaker.inputs import TrainingInput
    from sagemaker.pytorch import PyTorch

    # --------------------------------------------------------
    # SageMaker session
    # --------------------------------------------------------

    boto_session = boto3.Session(
        region_name=region,
    )

    sm_session = sagemaker.Session(
        boto_session=boto_session,
    )

    estimator_kwargs[
        "sagemaker_session"
    ] = sm_session

    # --------------------------------------------------------
    # Create estimator
    # --------------------------------------------------------

    estimator = PyTorch(
        **estimator_kwargs,
    )

    # --------------------------------------------------------
    # Convert configured channels to TrainingInput objects
    # --------------------------------------------------------

    training_inputs = {
        name: TrainingInput(
            s3_uri,
            input_mode=mode,
        )
        for name, (
            s3_uri,
            mode,
        ) in channels.items()
    }

    # --------------------------------------------------------
    # Submit training job
    # --------------------------------------------------------

    print(
        f"[launch] submitting: {run_name}"
    )

    estimator.fit(
        training_inputs,
        job_name=run_name,
        wait=not args.no_wait,
    )

    if args.no_wait:
        print(
            "[launch] submitted: "
            f"{estimator.latest_training_job.name}"
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()