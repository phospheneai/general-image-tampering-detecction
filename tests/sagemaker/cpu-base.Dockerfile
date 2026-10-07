# CPU stand-in for the AWS PyTorch DLC that sagemaker/Dockerfile builds FROM.
# Provides exactly what the DLC provides and requirements.txt deliberately does
# not pin (torch/torchvision + scikit-learn/seaborn/matplotlib/tqdm/pyyaml), at
# the DLC's versions, so CI can build the real Dockerfile on top of it:
#
#   docker build -f tests/sagemaker/cpu-base.Dockerfile -t forgery-cpu-base tests/sagemaker
#   docker build -f sagemaker/Dockerfile --build-arg BASE_IMAGE=forgery-cpu-base -t forgery-train:ci .
FROM python:3.11-slim

# libglib2.0 for opencv-python-headless; libgomp for torch's CPU kernels
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
      numpy==1.26.4 scikit-learn seaborn matplotlib tqdm pyyaml
