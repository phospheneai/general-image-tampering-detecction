from __future__ import annotations

from authgenforge import *

from authgenforge.utils.aug_utils import jpeg_compress


log = logging.getLogger(__name__)


# ============================================================================
# Spatial / paired transforms
# ============================================================================


class CapMegapixelsPair:
    """
    Limit very large images by cropping them to a maximum pixel budget.

    The exact same crop is applied to the image and the forgery mask.

    This follows the reference CapMegapixels behavior: the image is cropped
    rather than resized so that native pixel density is preserved.
    """

    def __init__(
        self,
        max_mp: float = 3.0,
        max_side: int = 8000,
        center: bool = False,
    ):
        self.max_pixels = int(max_mp * 1_000_000)
        self.max_side = max_side
        self.center = center

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
    ):

        w, h = image.size
        pixels = w * h

        scale_mp = (
            (self.max_pixels / pixels) ** 0.5
            if pixels > self.max_pixels
            else 1.0
        )

        scale_side = (
            self.max_side / max(w, h)
            if max(w, h) > self.max_side
            else 1.0
        )

        scale = min(
            scale_mp,
            scale_side,
        )

        if scale >= 1.0:
            return image, mask

        new_w = max(
            1,
            round(w * scale),
        )

        new_h = max(
            1,
            round(h * scale),
        )

        if self.center:

            image = T.CenterCrop(
                (new_h, new_w)
            )(image)

            mask = T.CenterCrop(
                (new_h, new_w)
            )(mask)

        else:

            top, left, crop_h, crop_w = (
                T.RandomCrop.get_params(
                    image,
                    output_size=(new_h, new_w),
                )
            )

            image = TF.crop(
                image,
                top,
                left,
                crop_h,
                crop_w,
            )

            mask = TF.crop(
                mask,
                top,
                left,
                crop_h,
                crop_w,
            )

        return image, mask


_RESAMPLING_METHODS = [
    Image.Resampling.NEAREST,
    Image.Resampling.BOX,
    Image.Resampling.BILINEAR,
    Image.Resampling.HAMMING,
    Image.Resampling.BICUBIC,
    Image.Resampling.LANCZOS,
]


class RandomResizePair:
    """
    Randomly resize an image and its mask using the same scale factor.

    The image uses the randomly selected interpolation method.

    The mask always uses nearest-neighbor interpolation so that binary
    forgery labels are not blurred.
    """

    _MIN_SOURCE_SIDE = 128

    def __init__(
        self,
        scale_range: tuple = (0.3, 2.0),
        p: float = 1.0,
    ):
        self.scale_range = scale_range
        self.p = p

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
    ):

        if random.random() > self.p:
            return image, mask

        w, h = image.size

        low, high = self.scale_range

        if min(w, h) < self._MIN_SOURCE_SIDE:
            low = max(low, 1.0)

        factor = random.uniform(
            low,
            high,
        )

        method = random.choice(
            _RESAMPLING_METHODS
        )

        new_w = max(
            1,
            int(w * factor),
        )

        new_h = max(
            1,
            int(h * factor),
        )

        image = image.resize(
            (new_w, new_h),
            method,
        )

        mask = mask.resize(
            (new_w, new_h),
            Image.Resampling.NEAREST,
        )

        return image, mask


class PadRandomCropPair:
    """
    Pad an image/mask pair if necessary and then take the same random crop
    from both.
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
    ):

        w, h = image.size

        pad_w = max(
            0,
            self.size - w,
        )

        pad_h = max(
            0,
            self.size - h,
        )

        if pad_w > 0 or pad_h > 0:

            padding = (
                pad_w // 2,
                pad_h // 2,
                pad_w - pad_w // 2,
                pad_h - pad_h // 2,
            )

            image = TF.pad(
                image,
                padding,
                fill=0,
            )

            mask = TF.pad(
                mask,
                padding,
                fill=0,
            )

        top, left, crop_h, crop_w = (
            T.RandomCrop.get_params(
                image,
                output_size=(
                    self.size,
                    self.size,
                ),
            )
        )

        image = TF.crop(
            image,
            top,
            left,
            crop_h,
            crop_w,
        )

        mask = TF.crop(
            mask,
            top,
            left,
            crop_h,
            crop_w,
        )

        return image, mask


class PadCenterCropPair:
    """
    Deterministic center crop for validation.

    The exact same crop is applied to the image and mask.
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
    ):

        w, h = image.size

        pad_w = max(
            0,
            self.size - w,
        )

        pad_h = max(
            0,
            self.size - h,
        )

        if pad_w > 0 or pad_h > 0:

            padding = (
                pad_w // 2,
                pad_h // 2,
                pad_w - pad_w // 2,
                pad_h - pad_h // 2,
            )

            image = TF.pad(
                image,
                padding,
                fill=0,
            )

            mask = TF.pad(
                mask,
                padding,
                fill=0,
            )

        image = T.CenterCrop(
            self.size
        )(image)

        mask = T.CenterCrop(
            self.size
        )(mask)

        return image, mask


# ============================================================================
# Image-only transformations
# ============================================================================


class RandomJPEGCompression:
    """
    Random JPEG compression applied only to the image.
    """

    def __init__(
        self,
        quality_lower: int = 55,
        quality_upper: int = 99,
        p: float = 0.3,
    ):
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper
        self.p = p

    def __call__(
        self,
        image: Image.Image,
    ):

        if random.random() < self.p:

            quality = random.randint(
                self.quality_lower,
                self.quality_upper,
            )

            return jpeg_compress(
                image,
                quality=quality,
            )

        return image


class RandomWebPCompression:
    """
    Random WebP compression applied only to the image.
    """

    def __init__(
        self,
        quality_range: tuple = (55, 100),
        p: float = 0.5,
    ):
        self.quality_range = quality_range
        self.p = p

    def _compress(
        self,
        image: Image.Image,
        quality: int,
    ):

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="WEBP",
            quality=quality,
        )

        buffer.seek(0)

        output = Image.open(
            buffer
        )

        output.load()

        if output.mode != image.mode:
            output = output.convert(
                image.mode
            )

        return output

    def __call__(
        self,
        image: Image.Image,
    ):

        if random.random() > self.p:
            return image

        quality = random.randint(
            *self.quality_range
        )

        return self._compress(
            image,
            quality,
        )


class RandomHeifCompression:
    """
    Random HEIF compression applied only to the image.

    pillow-heif is imported lazily.
    """

    def __init__(
        self,
        quality_range: tuple = (55, 100),
        p: float = 0.5,
    ):
        self.quality_range = quality_range
        self.p = p

    def _compress(
        self,
        image: Image.Image,
        quality: int,
    ):

        import pillow_heif

        pillow_heif.register_heif_opener()

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="HEIF",
            quality=quality,
        )

        buffer.seek(0)

        output = Image.open(
            buffer
        )

        output.load()

        return output

    def __call__(
        self,
        image: Image.Image,
    ):

        if random.random() > self.p:
            return image

        quality = random.randint(
            *self.quality_range
        )

        return self._compress(
            image,
            quality,
        )


class RandomDoubleCompression:
    """
    Apply one JPEG/WebP/HEIF compression pass and optionally apply
    a second JPEG compression pass.

    This follows the provenance-laundering logic from the reference repo.
    """

    def __init__(
        self,
        overall_p: float = 0.8,
        jpeg_quality: tuple = (55, 100),
        webp_quality: tuple = (55, 100),
        heif_quality: tuple = (55, 100),
        weights: tuple = (0.8, 0.2, 0.0),
        second_pass_p: float = 0.1,
        second_jpeg_quality: tuple = (55, 99),
    ):

        self.overall_p = overall_p

        self.jpeg = RandomJPEGCompression(
            quality_lower=jpeg_quality[0],
            quality_upper=jpeg_quality[1],
            p=1.0,
        )

        self.webp = RandomWebPCompression(
            quality_range=webp_quality,
            p=1.0,
        )

        self.heif = RandomHeifCompression(
            quality_range=heif_quality,
            p=1.0,
        )

        self.weights = weights

        self.second_pass_p = second_pass_p

        self.second_jpeg = RandomJPEGCompression(
            quality_lower=second_jpeg_quality[0],
            quality_upper=second_jpeg_quality[1],
            p=1.0,
        )

    def __call__(
        self,
        image: Image.Image,
    ):

        if random.random() > self.overall_p:
            return image

        transform = random.choices(
            [
                self.jpeg,
                self.webp,
                self.heif,
            ],
            weights=self.weights,
            k=1,
        )[0]

        image = transform(image)

        if random.random() < self.second_pass_p:

            image = self.second_jpeg(
                image
            )

        return image


class RandomCompression:
    """
    Apply one of JPEG, WebP, or HEIF compression.
    """

    def __init__(
        self,
        overall_p: float = 0.4,
        jpeg_quality: tuple = (55, 100),
        webp_quality: tuple = (55, 100),
        heif_quality: tuple = (55, 100),
        weights: tuple = (0.7, 0.2, 0.1),
    ):

        self.overall_p = overall_p

        self.jpeg = RandomJPEGCompression(
            quality_lower=jpeg_quality[0],
            quality_upper=jpeg_quality[1],
            p=1.0,
        )

        self.webp = RandomWebPCompression(
            quality_range=webp_quality,
            p=1.0,
        )

        self.heif = RandomHeifCompression(
            quality_range=heif_quality,
            p=1.0,
        )

        self.weights = weights

    def __call__(
        self,
        image: Image.Image,
    ):

        if random.random() > self.overall_p:
            return image

        transform = random.choices(
            [
                self.jpeg,
                self.webp,
                self.heif,
            ],
            weights=self.weights,
            k=1,
        )[0]

        return transform(image)


class NumpyBlock:
    """
    PIL -> NumPy -> NumPy transforms -> PIL.
    """

    def __init__(
        self,
        np_transforms: list,
    ):
        self.np_transforms = np_transforms

    def __call__(
        self,
        image: Image.Image,
    ):

        array = np.array(image)

        for transform in self.np_transforms:
            array = transform(array)

        return Image.fromarray(array)


# ============================================================================
# NumPy-native transformations
# ============================================================================


class MedianBlur:
    def __init__(
        self,
        kernel_size: int = 3,
    ):
        self.kernel_size = (
            kernel_size
            if kernel_size % 2 == 1
            else kernel_size + 1
        )

    def __call__(
        self,
        array: np.ndarray,
    ):

        return cv2.medianBlur(
            array,
            self.kernel_size,
        )


class MotionBlur:
    def __init__(
        self,
        kernel_size: int = 5,
        angle: float | None = None,
    ):

        self.kernel_size = (
            kernel_size
            if kernel_size % 2 == 1
            else kernel_size + 1
        )

        self.angle = angle

    def __call__(
        self,
        array: np.ndarray,
    ):

        angle = (
            self.angle
            if self.angle is not None
            else np.random.uniform(0, 180)
        )

        k = self.kernel_size
        center = k // 2

        kernel = np.zeros(
            (k, k),
            dtype=np.float32,
        )

        radians = np.deg2rad(
            angle
        )

        for i in range(k):

            x = int(
                center
                + np.round(
                    (i - center)
                    * np.cos(radians)
                )
            )

            y = int(
                center
                + np.round(
                    (i - center)
                    * np.sin(radians)
                )
            )

            if 0 <= x < k and 0 <= y < k:
                kernel[y, x] = 1

        kernel /= kernel.sum()

        if array.ndim == 3:

            output = np.stack(
                [
                    cv2.filter2D(
                        array[:, :, channel],
                        -1,
                        kernel,
                    )
                    for channel in range(
                        array.shape[2]
                    )
                ],
                axis=2,
            )

        else:

            output = cv2.filter2D(
                array,
                -1,
                kernel,
            )

        return output


class RandomSharpen:
    def __init__(
        self,
        p: float = 0.1,
        factor: tuple = (1.0, 3.0),
    ):

        self.p = p

        if isinstance(
            factor,
            (int, float),
        ):
            self.min_factor = factor
            self.max_factor = factor

        else:
            self.min_factor = factor[0]
            self.max_factor = factor[1]

    def __call__(
        self,
        array: np.ndarray,
    ):

        if random.random() > self.p:
            return array

        array_float = (
            array.astype(
                np.float32
            )
        )

        blurred = cv2.GaussianBlur(
            array_float,
            (0, 0),
            3.0,
        )

        factor = random.uniform(
            self.min_factor,
            self.max_factor,
        )

        sharpened = (
            array_float
            + factor
            * (array_float - blurred)
        )

        return np.clip(
            sharpened,
            0,
            255,
        ).astype(np.uint8)


class RandomPepperNoise:
    def __init__(
        self,
        p: float = 0.1,
        noise_ratio: tuple = (0.01, 0.10),
    ):

        self.p = p

        if isinstance(
            noise_ratio,
            (int, float),
        ):
            self.min_ratio = noise_ratio
            self.max_ratio = noise_ratio

        else:
            self.min_ratio = noise_ratio[0]
            self.max_ratio = noise_ratio[1]

    def __call__(
        self,
        array: np.ndarray,
    ):

        if random.random() > self.p:
            return array

        array = array.copy()

        h, w = array.shape[:2]

        ratio = random.uniform(
            self.min_ratio,
            self.max_ratio,
        )

        n = int(
            h * w * ratio
        )

        ys = np.random.randint(
            0,
            h,
            n,
        )

        xs = np.random.randint(
            0,
            w,
            n,
        )

        array[ys, xs] = 0

        return array


class RandomGaussianNoise:
    def __init__(
        self,
        mean: float = 0,
        std: float = 40,
        p: float = 0.5,
    ):

        self.mean = mean
        self.std = std
        self.p = p

    def __call__(
        self,
        array: np.ndarray,
    ):

        if random.random() < self.p:

            noisy = (
                array.astype(
                    np.float32
                )
                + np.random.normal(
                    self.mean,
                    self.std,
                    array.shape,
                )
            )

            return np.clip(
                noisy,
                0,
                255,
            ).astype(np.uint8)

        return array


# ============================================================================
# Pair-independent composition helpers
# ============================================================================


class PairRandomChoice:
    """
    Randomly select one paired transformation.
    """

    def __init__(
        self,
        transforms: list,
    ):
        self.transforms = transforms

    def __call__(
        self,
        image,
        mask,
    ):

        transform = random.choice(
            self.transforms
        )

        return transform(
            image,
            mask,
        )


class ImageRandomChoice:
    """
    Randomly select one image-only transformation.
    """

    def __init__(
        self,
        transforms: list,
    ):
        self.transforms = transforms

    def __call__(
        self,
        image,
    ):

        transform = random.choice(
            self.transforms
        )

        return transform(image)


# ============================================================================
# Tensor conversion
# ============================================================================


class ImageToTensor:
    """
    Convert image PIL data to a float tensor.

    Normalization is intentionally NOT performed here because DINOv3
    normalization is handled by the model's Norm module.
    """

    def __call__(
        self,
        image: Image.Image,
    ):

        return TF.to_tensor(
            image
        )


class MaskToTensor:
    """
    Convert a forgery mask into a binary float tensor.
    """

    def __call__(
        self,
        mask: Image.Image,
    ):

        tensor = TF.to_tensor(
            mask
        )

        return (
            tensor > 0.5
        ).float()