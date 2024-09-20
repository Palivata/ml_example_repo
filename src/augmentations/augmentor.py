import numpy as np

from src.augmentations.augmentations import (
    AVAILABLE_AUGMENTATIONS,
    AVAILABLE_PROCESSORS,
    Compose,
)


class ImageAugmentor(object):
    def __init__(self, configs: dict) -> None:
        self.augmentations = {}

        for stage, composers in configs.items():
            self.augmentations.setdefault(stage, [])
            for composer, augmentations in composers.items():
                composer = AVAILABLE_PROCESSORS[composer]
                transforms = []
                for augmentation_type, augmentation_params in augmentations.items():
                    aug = AVAILABLE_AUGMENTATIONS[augmentation_type](**augmentation_params)
                    transforms.append(aug)
                self.augmentations[stage].append(composer(transforms))
            self.augmentations[stage] = Compose(self.augmentations[stage])

    def __call__(self, image: np.ndarray, stage: str, **kwargs) -> np.ndarray:
        return self.augmentations[stage](image, **kwargs)
