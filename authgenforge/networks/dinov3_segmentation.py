
"""
DINOv3 ViT-L/16 + LoRA for general image forgery segmentation.

The backbone is loaded from a local Hugging Face model directory.

Expected backbone directory:
    config.json
    model.safetensors

Architecture:
    Image
      ↓
    ImageNet normalization
      ↓
    DINOv3 ViT-L/16
      ↓
    LoRA on Q/K/V projections
      ↓
    Dense patch features
      ↓
    3-convolution segmentation head
      ↓
    Full-resolution binary forgery mask logits

The pretrained DINOv3 weights are kept external to the Git repository
and can be supplied through an artifact location such as S3.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModel

from authgenforge import *


class Norm(nn.Module):
    """ImageNet normalization used before the DINOv3 backbone."""

    def __init__(self):
        super().__init__()

        self.register_buffer(
            "mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )

    def forward(self, x):
        return (x - self.mean) / self.std


class DINO(nn.Module):
    """
    Hugging Face DINOv3 ViT-L/16 backbone with LoRA.

    The backbone is loaded from a local Hugging Face directory.
    The directory must contain the DINOv3 config and pretrained
    model weights.

    Base DINOv3 parameters are frozen.
    Only LoRA parameters are trainable.
    """

    def __init__(
        self,
        backbone_path,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.0,
    ):
        super().__init__()

        self.backbone_path = backbone_path

        # ------------------------------------------------------------------
        # Load the Hugging Face configuration.
        # ------------------------------------------------------------------
        config = AutoConfig.from_pretrained(
            backbone_path,
            local_files_only=True,
        )

        # ------------------------------------------------------------------
        # Verify that this is the DINOv3 ViT-L/16 architecture expected
        # by this project.
        # ------------------------------------------------------------------
        expected = {
            "model_type": "dinov3_vit",
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "patch_size": 16,
            "num_register_tokens": 4,
        }

        for key, expected_value in expected.items():
            actual_value = getattr(config, key, None)

            if actual_value != expected_value:
                raise ValueError(
                    f"Unexpected DINOv3 configuration for '{key}': "
                    f"expected {expected_value}, got {actual_value}."
                )

        self.feature_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.num_register_tokens = config.num_register_tokens

        # ------------------------------------------------------------------
        # Load the pretrained Hugging Face DINOv3 weights.
        #
        # This reads model.safetensors from backbone_path.
        # No official DINOv3 GitHub repository is required.
        # ------------------------------------------------------------------
        self.dino = AutoModel.from_pretrained(
            backbone_path,
            config=config,
            local_files_only=True,
        )

        # ------------------------------------------------------------------
        # Freeze the complete pretrained DINOv3 backbone.
        # ------------------------------------------------------------------
        self.dino.requires_grad_(False)

        # ------------------------------------------------------------------
        # Apply LoRA to the Hugging Face representation of DINOv3 Q/K/V.
        #
        # Native DINOv3 uses a combined QKV projection, while the
        # Hugging Face implementation exposes q_proj/k_proj/v_proj.
        # ------------------------------------------------------------------
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
            ],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )

        self.dino = get_peft_model(
            self.dino,
            lora_config,
        )

        # ------------------------------------------------------------------
        # Make sure only LoRA parameters are trainable.
        # ------------------------------------------------------------------
        for name, param in self.dino.named_parameters():
            param.requires_grad = "lora_" in name

        trainable_params = sum(
            p.numel()
            for p in self.dino.parameters()
            if p.requires_grad
        )

        if trainable_params == 0:
            raise RuntimeError(
                "No trainable LoRA parameters were found. "
                "Check the Hugging Face DINOv3 module names and "
                "the installed PEFT/Transformers versions."
            )

    def forward(self, x):
        """
        Extract dense DINOv3 patch features.

        Input:
            (B, 3, H, W)

        Output:
            (B, 1024, H/16, W/16)
        """

        height, width = x.shape[-2:]

        if height % self.patch_size != 0:
            raise ValueError(
                f"Input height {height} must be divisible by "
                f"patch size {self.patch_size}."
            )

        if width % self.patch_size != 0:
            raise ValueError(
                f"Input width {width} must be divisible by "
                f"patch size {self.patch_size}."
            )

        outputs = self.dino(
            pixel_values=x,
            return_dict=True,
        )

        last_hidden = outputs.last_hidden_state

        # DINOv3 sequence:
        #
        #   CLS token
        #   Register tokens
        #   Patch tokens
        #
        # Remove CLS + four register tokens.
        patch_tokens = last_hidden[
            :,
            1 + self.num_register_tokens :,
        ]

        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        expected_tokens = num_patches_h * num_patches_w

        if patch_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                "Unexpected number of DINOv3 patch tokens: "
                f"got {patch_tokens.shape[1]}, "
                f"expected {expected_tokens}."
            )

        # Convert:
        #
        # (B, N, C)
        #
        # to:
        #
        # (B, C, H/16, W/16)
        #
        features = patch_tokens.reshape(
            patch_tokens.shape[0],
            num_patches_h,
            num_patches_w,
            self.feature_dim,
        )

        features = features.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

        return features


class SegmentationHead(nn.Module):
    """
    Three-convolution binary forgery segmentation head.

    DINOv3 ViT-L/16:
        1024 → 512 → 256 → 1
    """

    def __init__(self, feature_dim):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(
                feature_dim,
                feature_dim // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                feature_dim // 2,
                feature_dim // 4,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(feature_dim // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                feature_dim // 4,
                1,
                kernel_size=1,
            ),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                )

                if module.bias is not None:
                    nn.init.constant_(
                        module.bias,
                        0,
                    )

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(
                    module.weight,
                    1,
                )

                nn.init.constant_(
                    module.bias,
                    0,
                )

    def forward(self, features):
        return self.head(features)


class DINOv3Segmentation(nn.Module):
    """
    DINOv3 ViT-L/16 + LoRA + segmentation head.
    """

    def __init__(
        self,
        backbone_path,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.0,
    ):
        super().__init__()

        self.norm = Norm()

        self.backbone = DINO(
            backbone_path=backbone_path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        self.seg_head = SegmentationHead(
            feature_dim=self.backbone.feature_dim,
        )

    def forward_features(self, x):
        """Extract dense DINOv3 patch features."""

        x = self.norm(x)

        return self.backbone(x)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x:
                Input image tensor of shape (B, 3, H, W).

        Returns:
            Binary forgery mask logits of shape (B, 1, H, W).
        """

        input_size = x.shape[-2:]

        features = self.forward_features(x)

        logits = self.seg_head(features)

        # Restore prediction to the original image resolution.
        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits

    @torch.no_grad()
    def predict(self, x):
        """
        Inference helper.

        Returns:
            Sigmoid probability mask of shape (B, 1, H, W).
        """

        self.eval()

        logits = self.forward(x)

        return torch.sigmoid(logits)


def build_dinov3_segmentation(
    backbone_path: str,
    lora_rank: int = 32,
    lora_alpha: float = 64,
    lora_dropout: float = 0.0,
) -> DINOv3Segmentation:
    """
    Build the DINOv3 ViT-L/16 + LoRA segmentation model.

    backbone_path should point to a local Hugging Face model directory
    containing config.json and model.safetensors.
    """

    return DINOv3Segmentation(
        backbone_path=backbone_path,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )


def load_checkpoint(
    model: DINOv3Segmentation,
    checkpoint_path: str,
    strict: bool = True,
) -> DINOv3Segmentation:
    """
    Load a saved Authenta model checkpoint.

    Supports:
        - raw state_dict
        - {'model_state_dict': ...}
        - {'model': ...}
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

    elif (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
    ):
        state_dict = checkpoint["model"]

    else:
        state_dict = checkpoint

    # Remove DistributedDataParallel prefix if present.
    state_dict = {
        key.replace("module.", "", 1): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(
        state_dict,
        strict=strict,
    )

    if isinstance(checkpoint, dict):
        epoch = checkpoint.get("epoch", "?")
        global_step = checkpoint.get(
            "global_step",
            "?",
        )

        print(
            f"Loaded — epoch {epoch}"
            + (
                f", step {global_step:,}"
                if isinstance(global_step, int)
                else ""
            )
        )

    return model


if __name__ == "__main__":
    _HERE = os.path.dirname(
        os.path.abspath(__file__)
    )

    BACKBONE_PATH = os.path.join(
        _HERE,
        "..",
        "..",
        "artifacts",
        "mirror",
        "dinov3-vitl16",
    )

    model = build_dinov3_segmentation(
        backbone_path=BACKBONE_PATH,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.0,
    )

    model.eval()

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Backbone: DINOv3 ViT-L/16"
    )

    print(
        f"Feature dim: "
        f"{model.backbone.feature_dim}"
    )

    print(
        f"Total params: "
        f"{total_params / 1e6:.2f}M"
    )

    print(
        f"Trainable params: "
        f"{trainable_params / 1e6:.2f}M"
    )

    dummy = torch.randn(
        1,
        3,
        512,
        512,
    )

    with torch.no_grad():
        output = model(dummy)

    print(
        f"Output shape: "
        f"{tuple(output.shape)}"
    )
