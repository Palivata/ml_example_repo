import logging
import os
from typing import Optional

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler

from src.augmentations.augmentor import ImageAugmentor
from src.config import Config
from src.dataset_splitter import stratify_shuffle_split_subsets
from src.datasets import AVAILABLE_DATASETS


class DM(LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._batch_size = config.batch_size
        self._num_iterations = config.num_iterations
        self._n_workers = config.n_workers
        self._train_size = config.train_size
        self._data_path = config.data_path
        self.augmentor = ImageAugmentor(config.augmentations)
        self._image_folder = config.data_path

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.train_sampler: Optional[Sampler] = None
        self.common_dataset = AVAILABLE_DATASETS[config.experiment]

    def prepare_data(self):
        split_and_save_datasets(self._data_path, self._train_size)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            df_train = read_df(self._data_path, "train")
            df_valid = read_df(self._data_path, "valid")
            self.train_dataset = self.common_dataset(
                df_train, image_folder=self._image_folder, augmentor=self.augmentor, stage="train"
            )
            self.valid_dataset = self.common_dataset(
                df_valid, image_folder=self._image_folder, augmentor=self.augmentor, stage="test"
            )

            if self._num_iterations != -1:
                self.train_sampler = RandomSampler(
                    data_source=self.train_dataset,
                    num_samples=self._num_iterations * self._batch_size,
                )

        elif stage == "test":
            df_test = read_df(self._data_path, "test")
            self.test_dataset = self.common_dataset(
                df_test, image_folder=self._image_folder, augmentor=self.augmentor, stage="test"
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=False if self.train_sampler else True,
            sampler=self.train_sampler,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )


def split_and_save_datasets(data_path: str, train_fraction: float = 0.8):
    df = pd.read_csv(os.path.join(data_path, "train.csv"))
    logging.info(f"Original dataset: {len(df)}")
    df = df.drop_duplicates()
    logging.info(f"Final dataset: {len(df)}")

    train_df, valid_df, test_df = stratify_shuffle_split_subsets(df, train_fraction=train_fraction)
    logging.info(f"Train dataset: {len(train_df)}")
    logging.info(f"Valid dataset: {len(valid_df)}")
    logging.info(f"Test dataset: {len(test_df)}")

    train_df.to_csv(os.path.join(data_path, "df_train.csv"), index=False)
    valid_df.to_csv(os.path.join(data_path, "df_valid.csv"), index=False)
    test_df.to_csv(os.path.join(data_path, "df_test.csv"), index=False)
    logging.info("Datasets successfully saved!")


def read_df(data_path: str, mode: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_path, f"df_{mode}.csv"))
