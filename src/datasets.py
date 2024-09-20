import os

import albumentations as alb
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from numpy import ndarray
from torch.utils.data import Dataset

from src.augmentations.augmentor import ImageAugmentor

AVAILABLE_DATASETS = dict()


def register_dataset(name):
    def decorator(f):
        global AVAILABLE_DATASETS
        AVAILABLE_DATASETS[name] = f
        return f

    return decorator


@register_dataset("classification")
class ClassificationDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, image_folder: str, augmentor: ImageAugmentor, stage: str = "train"
    ):
        self.df = df
        self.image_folder = image_folder
        self.augmentor = augmentor
        self.stage = stage
        self.normalization = alb.Normalize()
        self.to_tensor = ToTensorV2()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ndarray[int]]:
        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_folder, f"{row.Id}.jpg")
        labels = np.array(row.drop(["Id"]), dtype="float32")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        data = {"image": image, "labels": labels, "stage": self.stage}

        augmented_image = self.augmentor(**data)["image"]
        augmented_image = self.normalization.apply(image=augmented_image)
        augmented_image = self.to_tensor.apply(img=augmented_image)
        return (augmented_image, data["labels"])

    def __len__(self) -> int:
        return len(self.df)


@register_dataset("segmentation")
class SegmentationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_folder: str,
        augmentor: alb.Compose,
        stage: str = "train",
    ):
        self.df = df
        self.image_folder = image_folder
        self.augmentor = augmentor
        self.stage = stage
        self.normalization = alb.Normalize()
        self.to_tensor = ToTensorV2()

    @staticmethod
    def boxes_to_mask(boxes: list, image_shape: np.ndarray) -> np.ndarray:
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        x_from, y_from, width, height = boxes
        x_min = x_from
        y_min = y_from
        x_max = x_from + width
        y_max = y_from + height
        mask[y_min:y_max, x_min:x_max] = 1
        return mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_folder, f"{row.filename}")
        boxes = row[["x_from", "y_from", "width", "height"]].tolist()

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self.boxes_to_mask(boxes, image_shape=image.shape)

        data = {"image": image, "mask": mask, "stage": self.stage}

        augmented = self.augmentor(**data)
        augmented_image = augmented["image"]
        augmented_mask = augmented["mask"]

        normalized_image = self.normalization(image=augmented_image)["image"]
        image_tensor = self.to_tensor(image=normalized_image)["image"]
        mask_tensor = torch.tensor(augmented_mask, dtype=torch.long)

        return image_tensor, mask_tensor

    def __len__(self) -> int:
        return len(self.df)


@register_dataset("ocr")
class OCRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_folder: str, augmentor=None, stage: str = "train"):
        self.df = df
        self.image_folder = image_folder
        self.augmentor = augmentor
        self.stage = stage
        self.normalization = alb.Normalize()
        self.to_tensor = ToTensorV2()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_folder, f"{row.filename}")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x1 = int(row["x_from"])
        y1 = int(row["y_from"])
        x2 = int(row["x_from"]) + int(row["width"])
        y2 = int(row["y_from"]) + int(row["height"])
        crop = image[y1:y2, x1:x2]

        if crop.shape[0] > crop.shape[1]:
            crop = cv2.rotate(crop, 2)

        text_label = str(row["code"])
        data = {
            "image": crop,
            "text": text_label,
            "text_length": len(text_label),
            "stage": self.stage,
        }
        augmented = self.augmentor(**data)
        augmented_image = augmented["image"]
        text_label = augmented["text"]

        normalized_image = self.normalization(image=augmented_image)["image"]
        image_tensor = self.to_tensor(image=normalized_image)["image"]
        text_label = torch.IntTensor(text_label)

        return image_tensor, text_label, len(text_label)

    def __len__(self) -> int:
        return len(self.df)
