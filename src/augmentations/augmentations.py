import albumentations as alb
import cv2
import numpy as np
from pydantic import BaseModel, validator


class Compose:
    def __init__(self, augmentations: list) -> None:
        self.augmentations = augmentations

    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        kwargs = {"image": image, **kwargs}
        for aug in self.augmentations:
            kwargs = aug(**kwargs)
        return kwargs


class OneOf:
    def __init__(self, augmentations: list) -> None:
        self.augmentations = augmentations

    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        kwargs = np.random.choice(self.augmentations)(image=image, **kwargs)
        return kwargs


AVAILABLE_AUGMENTATIONS = dict()
AVAILABLE_PROCESSORS = {"Compose": Compose, "OneOf": OneOf}


def register_augmentation(name):
    def decorator(f):
        global AVAILABLE_AUGMENTATIONS
        AVAILABLE_AUGMENTATIONS[name] = f
        return f

    return decorator


class BaseAugmentation(BaseModel):
    p: float

    @validator("p")
    def validate_p(cls, v):
        assert 0 <= v <= 1, "p must be between 0 and 1"
        return v

    def prob(self):
        return np.random.uniform() <= self.p


@register_augmentation("resize")
class Resize(BaseAugmentation):
    p: float
    height: int
    width: int
    resize_method: str

    @validator("resize_method")
    def validate_resize_method(cls, resize_method):
        assert resize_method in ["nearest", "bilinear", "bicubic"], (
            "resize_method must be either bilinear, bicubic " "or nearest"
        )
        if resize_method == "bilinear":
            return cv2.INTER_NEAREST
        elif resize_method == "bicubic":
            return cv2.INTER_CUBIC
        else:
            return cv2.INTER_NEAREST

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        mask = kwargs.get("mask", None)
        if self.prob():
            image = alb.resize(image, self.height, self.width, self.resize_method)
            if mask is not None:
                kwargs["mask"] = alb.resize(mask, self.height, self.width, cv2.INTER_NEAREST)

        return {"image": image, **kwargs}


@register_augmentation("vertical_flip")
class VerticalFlip(BaseAugmentation):
    p: float

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        mask = kwargs.get("mask", None)
        if self.prob():
            image = cv2.flip(image, 0)
            if mask is not None:
                kwargs["mask"] = cv2.flip(mask, 0)

        return {"image": image, **kwargs}


@register_augmentation("horisontal_flip")
class HorisontalFlip(BaseAugmentation):
    p: float

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        mask = kwargs.get("mask", None)
        if self.prob():
            if mask is not None:
                kwargs["mask"] = cv2.flip(mask, 1)
            image = cv2.flip(image, 1)

        return {"image": image, **kwargs}


@register_augmentation("color_jitter")
class ColorJitter(BaseAugmentation):
    p: float

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        if self.prob():
            image = alb.brightness_contrast_adjust(image)
        return {"image": image, **kwargs}


@register_augmentation("random_scaling")
class RandomScale(BaseAugmentation):
    p: float
    angle_low: float
    angle_max: float
    scale_low: float
    scale_max: float

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        mask = kwargs.get("mask", None)
        if self.prob():
            angle = np.random.uniform(self.angle_low, self.angle_max)
            scale = np.random.uniform(self.scale_low, self.scale_max)
            if mask is not None:
                kwargs["mask"] = alb.shift_scale_rotate(mask, angle, scale, 1, 1)
            image = alb.shift_scale_rotate(image, angle, scale, 1, 1)
        return {"image": image, **kwargs}


@register_augmentation("random_gaussian_blur")
class RandomGaussianBlur(BaseAugmentation):
    p: float
    kernel_low: int
    kernel_max: int

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        if self.prob():
            kernel = np.random.randint(self.kernel_low, self.kernel_max)
            kernel = kernel + kernel % 2 - 1
            image = alb.gaussian_blur(image, kernel)
        return {"image": image, **kwargs}


@register_augmentation("discrete_flip_rotate")
class RandomDiscreteRotate(BaseAugmentation):
    p: float

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        mask = kwargs.get("mask", None)
        if self.prob():
            choice = np.random.randint(0, 6)
            if choice == 0:
                image = cv2.rotate(image, rotateCode=cv2.ROTATE_90_CLOCKWISE)
                if mask is not None:
                    kwargs["mask"] = cv2.rotate(mask, rotateCode=cv2.ROTATE_90_CLOCKWISE)
            if choice == 1:
                image = cv2.rotate(image, rotateCode=cv2.ROTATE_90_CLOCKWISE)
                image = cv2.flip(image, flipCode=1)
                if mask is not None:
                    mask = cv2.rotate(mask, rotateCode=cv2.ROTATE_90_CLOCKWISE)
                    kwargs["mask"] = cv2.flip(mask, flipCode=1)
            if choice == 2:
                image = cv2.rotate(image, rotateCode=cv2.ROTATE_180)
                if mask is not None:
                    kwargs["mask"] = cv2.rotate(mask, rotateCode=cv2.ROTATE_180)
            if choice == 3:
                image = cv2.rotate(image, rotateCode=cv2.ROTATE_180)
                image = cv2.flip(image, flipCode=1)
                if mask is not None:
                    mask = cv2.rotate(mask, rotateCode=cv2.ROTATE_180)
                    kwargs["mask"] = cv2.flip(mask, flipCode=1)
            if choice == 4:
                image = cv2.rotate(image, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
                if mask is not None:
                    kwargs["mask"] = cv2.rotate(mask, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
            if choice == 5:
                image = cv2.rotate(image, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
                image = cv2.flip(image, flipCode=1)
                if mask is not None:
                    mask = cv2.rotate(mask, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
                    kwargs["mask"] = cv2.flip(mask, flipCode=1)
        return {"image": image, **kwargs}


@register_augmentation("elastic_transform")
class ElasticTransform(BaseAugmentation):
    p: float
    alpha: float
    sigma: float
    alpha_affine: float

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        mask = kwargs.get("mask", None)
        if self.prob():
            elastic_transform = alb.ElasticTransform(
                alpha=self.alpha, sigma=self.sigma, alpha_affine=self.alpha_affine, p=1.0
            )
            augmented = elastic_transform(image=image, mask=mask)
            image = augmented["image"]
            if mask is not None:
                kwargs["mask"] = augmented["mask"]
        return {"image": image, **kwargs}


@register_augmentation("grid_distortion")
class GridDistortion(BaseAugmentation):
    p: float
    num_steps: int
    distort_limit: float

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        mask = kwargs.get("mask", None)
        if self.prob():
            grid_distortion = alb.GridDistortion(
                num_steps=self.num_steps, distort_limit=self.distort_limit, p=1.0
            )
            augmented = grid_distortion(image=image, mask=mask)
            image = augmented["image"]
            if mask is not None:
                kwargs["mask"] = augmented["mask"]
        return {"image": image, **kwargs}


@register_augmentation("ocr_resize")
class PadResizeOCR(BaseAugmentation):
    target_width: int
    target_height: int
    mode: str
    p: float

    @validator("mode")
    def validate_mode_method(cls, mode):
        assert mode in ["random", "left", "center"]

    def __call__(self, image, **kwargs) -> dict:
        h, w = image.shape[:2]

        tmp_w = min(int(w * (self.target_height / h)), self.target_width)
        image = cv2.resize(image, (tmp_w, self.target_height))

        dw = np.round(self.target_width - tmp_w).astype(int)
        if dw > 0:
            if self.mode == "random":
                pad_left = np.random.randint(dw)
            elif self.mode == "left":
                pad_left = 0
            else:
                pad_left = dw // 2

            pad_right = dw - pad_left

            image = cv2.copyMakeBorder(
                image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
            )

        return {"image": image, **kwargs}


@register_augmentation("text_encoding")
class TextEncode(BaseAugmentation):
    vocab: str
    target_text_size: int
    p: float

    @validator("vocab")
    def validate_vocab_method(cls, vocab):
        vocab = list(vocab)
        return vocab

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        source_text = kwargs["text"].strip()

        postprocessed_text = [self.vocab.index(x) + 1 for x in source_text if x in self.vocab]
        postprocessed_text = np.pad(
            postprocessed_text,
            (0, self.target_text_size - len(postprocessed_text)),
            mode="constant",
        )

        kwargs["text"] = postprocessed_text

        return {"image": image, **kwargs}


@register_augmentation("ocr_perspective_crop")
class CropPerspective(BaseAugmentation):
    p: float
    width_ratio: float
    height_ratio: float

    def __call__(self, image: np.ndarray, **kwargs):

        if self.prob():
            h, w, c = image.shape

            pts1 = np.float32([[0, 0], [0, h], [w, h], [w, 0]])
            dw = w * self.width_ratio
            dh = h * self.height_ratio

            pts2 = np.float32(
                [
                    [np.random.uniform(-dw, dw), np.random.uniform(-dh, dh)],
                    [np.random.uniform(-dw, dw), h - np.random.uniform(-dh, dh)],
                    [w - np.random.uniform(-dw, dw), h - np.random.uniform(-dh, dh)],
                    [w - np.random.uniform(-dw, dw), np.random.uniform(-dh, dh)],
                ]
            )

            matrix = cv2.getPerspectiveTransform(pts2, pts1)
            dst_w = (pts2[3][0] + pts2[2][0] - pts2[1][0] - pts2[0][0]) * 0.5
            dst_h = (pts2[2][1] + pts2[1][1] - pts2[3][1] - pts2[0][1]) * 0.5
            image = cv2.warpPerspective(
                image,
                matrix,
                (int(dst_w), int(dst_h)),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        return {"image": image, **kwargs}


@register_augmentation("x_scale")
class ScaleX(BaseAugmentation):
    p: float
    scale_min: float
    scale_max: float

    def __call__(self, image: np.ndarray, **kwargs):

        if self.prob():
            h, w, c = image.shape
            w = int(w * np.random.uniform(self.scale_min, self.scale_max))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        return {"image": image, **kwargs}
