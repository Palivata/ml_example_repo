from typing import List

from omegaconf import OmegaConf
from pydantic import BaseModel


class LossConfig(BaseModel):
    """Configuration class for loss functions"""

    name: str
    weight: float
    loss_fn: str
    loss_kwargs: dict


class Config(BaseModel):
    """Configuration class for model configurations"""

    project_name: str
    experiment_name: str
    experiment: str
    data_path: str
    batch_size: int
    n_workers: int
    num_iterations: int
    train_size: float
    width: int
    height: int
    n_epochs: int
    num_classes: int
    accelerator: str
    device: int
    monitor_metric: str
    monitor_mode: str
    model_kwargs: dict
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict
    losses: List[LossConfig]
    augmentations: dict
    seed: int

    @classmethod
    def from_yaml(cls, path: str) -> "Config":

        """
        Load configuration from YAML file
        - path - path to YAML file
        - Returns Configuration object
        """

        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
