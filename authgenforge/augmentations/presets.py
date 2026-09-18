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


class _GaussianBlur:
    """
    Picklable stand-in for `lambda image: TF.gaussian_blur(image, kernel_size=k)`.

    DataLoader workers on Windows/macOS (spawn start method, unlike Linux's
    fork) pickle the transform to send it to each worker process — a local
    lambda/closure can't be pickled, so it must be a plain class instance.
    """

    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size

    def __call__(self, image):
        return TF.gaussian_blur(
            image,
            kernel_size=self.kernel_size,
        )


class _TrainTransform:
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

    A class (not a closure) so DataLoader worker processes can pickle it —
    see _GaussianBlur's docstring.
    """

    def __init__(
        self,
        crop_size: int = 512,
    ):

        self.crop_size = crop_size

        self.cap = CapMegapixelsPair(
            max_mp=3.0,
            max_side=8000,
            center=False,
        )

        self.resize = RandomResizePair(
            scale_range=(0.3, 2.0),
            p=0.3,
        )

        self.noise_choice = ImageRandomChoice(
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

        self.sharpen = NumpyBlock(
            [
                RandomSharpen(
                    p=0.1
                )
            ]
        )

        self.compression = RandomDoubleCompression(
            overall_p=1.0,
            weights=(0.8, 0.2, 0.0),
            second_pass_p=0.2,
            second_jpeg_quality=(55, 99),
        )

        self.blur_choice = ImageRandomChoice(
            [
                _GaussianBlur(kernel_size=3),
                _GaussianBlur(kernel_size=5),
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

        self.crop = PadRandomCropPair(
            size=crop_size
        )

        self.image_to_tensor = ImageToTensor()
        self.mask_to_tensor = MaskToTensor()

    def __call__(
        self,
        image,
        mask,
    ):

        # --------------------------------------------------------------
        # Bound expensive operations
        # --------------------------------------------------------------

        image, mask = self.cap(
            image,
            mask,
        )

        # --------------------------------------------------------------
        # Random resize
        # --------------------------------------------------------------

        image, mask = self.resize(
            image,
            mask,
        )

        # --------------------------------------------------------------
        # Noise
        # --------------------------------------------------------------

        image = self.noise_choice(
            image
        )

        # --------------------------------------------------------------
        # Sharpening
        # --------------------------------------------------------------

        image = self.sharpen(
            image
        )

        # --------------------------------------------------------------
        # Compression
        # --------------------------------------------------------------

        if random.random() < 0.5:

            image = self.compression(
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

            image = self.blur_choice(
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

        image, mask = self.crop(
            image,
            mask,
        )

        # --------------------------------------------------------------
        # Tensor conversion
        # --------------------------------------------------------------

        image = self.image_to_tensor(
            image
        )

        mask = self.mask_to_tensor(
            mask
        )

        return image, mask


def get_train_transforms(
    crop_size: int = 512,
):
    return _TrainTransform(crop_size=crop_size)


# ============================================================================
# Validation
# ============================================================================


class _ValTransform:
    """
    Deterministic validation transform.

    No random augmentation is applied.

    The same deterministic spatial transformation is applied to the
    image and mask.

    A class (not a closure) so DataLoader worker processes can pickle
    it — see _GaussianBlur's docstring above.
    """

    def __init__(
        self,
        crop_size: int = 512,
    ):

        self.cap = CapMegapixelsPair(
            max_mp=3.0,
            max_side=8000,
            center=True,
        )

        self.crop = PadCenterCropPair(
            size=crop_size
        )

        self.image_to_tensor = ImageToTensor()
        self.mask_to_tensor = MaskToTensor()

    def __call__(
        self,
        image,
        mask,
    ):

        # --------------------------------------------------------------
        # Deterministic megapixel cap
        # --------------------------------------------------------------

        image, mask = self.cap(
            image,
            mask,
        )

        # --------------------------------------------------------------
        # Deterministic center crop
        # --------------------------------------------------------------

        image, mask = self.crop(
            image,
            mask,
        )

        # --------------------------------------------------------------
        # Tensor conversion
        # --------------------------------------------------------------

        image = self.image_to_tensor(
            image
        )

        mask = self.mask_to_tensor(
            mask
        )

        return image, mask


def get_val_transforms(
    crop_size: int = 512,
):
    return _ValTransform(crop_size=crop_size)