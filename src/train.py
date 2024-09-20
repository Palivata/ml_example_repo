import argparse
import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
from clearml import Task
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src.config import Config
from src.constants import EXPERIMENTS_PATH
from src.datamodule import DM
from src.module import AVAILABLE_MODULES


def train(config: Config):
    model = AVAILABLE_MODULES[config.experiment](config)
    datamodule = DM(config)
    task = Task.init(
        project_name=config.project_name,
        task_name=f"{config.experiment_name}",
        auto_connect_frameworks=True,
    )
    task.connect(config.dict())

    experiment_save_path = os.path.join(EXPERIMENTS_PATH, config.experiment_name)
    os.makedirs(experiment_save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        experiment_save_path,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=1,
        filename=f"epoch_{{epoch:02d}}-{{{config.monitor_metric}:.3f}}",
    )
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=[config.device],
        log_every_n_steps=20,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor=config.monitor_metric, patience=4, mode=config.monitor_mode),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="config file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = Config.from_yaml(args.config_file)
    pl.seed_everything(config.seed, workers=True)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    train(config)
