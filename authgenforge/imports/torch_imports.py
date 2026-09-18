from __future__ import annotations

import torch

torch.set_num_threads(1)  # default (= cpu count) makes tiny per-image ops (ToTensor, normalize) pay huge thread-dispatch overhead

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as T
import torchvision.transforms.functional as TF

from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoModel

from peft import get_peft_model, LoraConfig