from __future__ import annotations

from authgenforge import *

from authgenforge.augmentations.blocks import (
    CapMegapixelsPair,
    RandomResizePair,
    PadRandomCropPair,
    PadCenterCropPair,
    RandomJPEGCompression,
    RandomWebPCompression,
    RandomHeifCompression,
    RandomDoubleCompression,
    NumpyBlock,
    MedianBlur,
    MotionBlur,
    RandomSharpen,
    RandomPepperNoise,
    RandomGaussianNoise,
    PairRandomChoice,
    ImageRandomChoice,
    ImageToTensor,
    MaskToTensor,
)


# ============================================================================
# Training
# ============================================================================


def get_train_transforms(
    crop_size: int = 512,
):

    """
    Default training augmentation preset for general image forgery
    segmentation.

    Spatial transformations are synchronized between image and mask.

    Image-only provenance transformations include:
        - resize
        - noise
        - sharpening
        - JPEG/WebP/HEIF compression
        - double compression
        - blur
        - color jitter
        - grayscale
    """

    cap = CapMegapixelsPair(
        max_mp=3.0,
        max_side=8000,
        center=False,
    )

    resize = RandomResizePair(
        scale_range=(0.3, 2.0),
        p=0.3,
    )

    noise_choice = ImageRandomChoice(
        [
            NumpyBlock(
                [
                    RandomGaussianNoise(
                        p=0.1
                    )
                ]
            ),
            NumpyBlock(
                [
                    RandomPepperNoise(
                        p=0.1
                    )
                ]
            ),
        ]
    )

    sharpen = NumpyBlock(
        [
            RandomSharpen(
                p=0.1
            )
        ]
    )

    compression = RandomDoubleCompression(
        overall_p=1.0,
        weights=(0.8, 0.2, 0.0),
        second_pass_p=0.2,
        second_jpeg_quality=(55, 99),
    )

    blur_choice = ImageRandomChoice(
        [
            lambda image: TF.gaussian_blur(
                image,
                kernel_size=3,
            ),
            lambda image: TF.gaussian_blur(
                image,
                kernel_size=5,
            ),
            NumpyBlock(
                [
                    MedianBlur(
                        kernel_size=3
                    )
                ]
            ),
            NumpyBlock(
                [
                    MotionBlur(
                        kernel_size=5
                    )
                ]
            ),
        ]
    )

    image_to_tensor = ImageToTensor()
    mask_to_tensor = MaskToTensor()

    def transform(
        image,
        mask,
    ):

        # --------------------------------------------------------------
        # Bound expensive operations
        # --------------------------------------------------------------

        image, mask = cap(
            image,
            mask,
        )

        # --------------------------------------------------------------
        # Random resize
        # --------------------------------------------------------------

        image, mask = resize(
            image,
            mask,
        )

        # --------------------------------------------------------------
        # Noise
        # --------------------------------------------------------------

        image = noise_choice(
            image
        )

        # --------------------------------------------------------------
        # Sharpening
        # --------------------------------------------------------------

        image = sharpen(
            image
        )

        # --------------------------------------------------------------
        # Compression
        # --------------------------------------------------------------

        if random.random() < 0.5:

            image = compression(
                image
            )

        # --------------------------------------------------------------
        # Geometric flips
        # --------------------------------------------------------------

        if random.random() < 0.5:

            image = TF.hflip(
                image
            )

            mask = TF.hflip(
                mask
            )

        if random.random() < 0.5:

            image = TF.vflip(
                image
            )

            mask = TF.vflip(
                mask
            )

        # --------------------------------------------------------------
        # Blur
        # --------------------------------------------------------------

        if random.random() < 0.2:

            image = blur_choice(
                image
            )

        # --------------------------------------------------------------
        # Color jitter
        # --------------------------------------------------------------

        if random.random() < 0.2:

            image = T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.175,
            )(image)

        # --------------------------------------------------------------
        # Grayscale
        # --------------------------------------------------------------

        if random.random() < 0.1:

            image = T.Grayscale(
                num_output_channels=3
            )(image)

        # --------------------------------------------------------------
        # Final crop
        # --------------------------------------------------------------

        image, mask = PadRandomCropPair(
            size=crop_size
        )(
            image,
            mask,
        )

        # --------------------------------------------------------------
        # Tensor conversion
        # --------------------------------------------------------------

        image = image_to_tensor(
            image
        )

        mask = mask_to_tensor(
            mask
        )

        return image, mask

    return transform


# ============================================================================
# Validation
# ============================================================================


def get_val_transforms(
    crop_size: int = 512,
):

    """
    Deterministic validation transform.

    No random augmentation is applied.

    The same deterministic spatial transformation is applied to the
    image and mask.
    """

    cap = CapMegapixelsPair(
        max_mp=3.0,
        max_side=8000,
        center=True,
    )

    crop = PadCenterCropPair(
        size=crop_size
    )

    image_to_tensor = ImageToTensor()
    mask_to_tensor = MaskToTensor()

    def transform(
        image,
        mask,
    ):

        # --------------------------------------------------------------
        # Deterministic megapixel cap
        # --------------------------------------------------------------

        image, mask = cap(
            image,
            mask,
        )

        # --------------------------------------------------------------
        # Deterministic center crop
        # --------------------------------------------------------------

        image, mask = crop(
            image,
            mask,
        )

        # --------------------------------------------------------------
        # Tensor conversion
        # --------------------------------------------------------------

        image = image_to_tensor(
            image
        )

        mask = mask_to_tensor(
            mask
        )

        return image, mask

    return transform