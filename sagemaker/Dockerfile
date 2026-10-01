# Forgery-segmentation training image. Build context is the REPO ROOT:
#   docker build -f sagemaker/Dockerfile -t forgery-train:dev .
#
# Base is the AWS Deep Learning Container launch.py pins (framework 2.3 /
# py311), so requirements.txt stays valid as-is: the DLC supplies torch/
# torchvision/cuda + scikit-learn/seaborn/matplotlib/tqdm/pyyaml, and
# requirements.txt pins only what we add. It lives in AWS's DLC account, so
# the build needs `aws ecr get-login-password` against 763104351884 first
# (the Build training image workflow does that).
#
# BASE_IMAGE is overridable so CI can build the exact same layers on a
# CPU-only base (tests/sagemaker/cpu-base.Dockerfile) without AWS access.
ARG BASE_IMAGE=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker
FROM ${BASE_IMAGE}

ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA} \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/ml/code

# HF must not write to $HOME (not writable on a training instance); /tmp is.
ENV HF_HOME=/tmp/hf \
    HF_DATASETS_CACHE=/tmp/hf/datasets \
    TQDM_MININTERVAL=60

WORKDIR /opt/ml/code

# deps first — this layer is cached until requirements.txt changes, so a code
# push reuses the (large) pip layer instead of reinstalling everything.
COPY sagemaker/requirements.txt /opt/ml/code/sagemaker/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/code/sagemaker/requirements.txt

# Only what training needs, laid out as in the repo so relative paths resolve
# identically:
#   authgenforge/  the trainer package (importable via PYTHONPATH=/opt/ml/code)
#   sagemaker/     train.py + config/ + backbone/
# The DINOv3 weights are deliberately absent — they arrive on the `artifacts`
# S3 channel at /opt/ml/input/data/artifacts/.
COPY authgenforge/ /opt/ml/code/authgenforge/
COPY sagemaker/    /opt/ml/code/sagemaker/

# The state machines set ContainerEntrypoint/ContainerArguments explicitly;
# this default only matters for a bare `docker run`. train.py uses
# parse_known_args, so the legacy `train` argument SageMaker can append is
# dropped harmlessly.
ENTRYPOINT ["python", "/opt/ml/code/sagemaker/train.py"]
CMD ["--config", "config/normal/train_forensics.yml"]
