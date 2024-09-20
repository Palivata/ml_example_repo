import pytorch_lightning as pl
import torch
from segmentation_models_pytorch import Unet
from timm import create_model

from src.config import Config
from src.losses import get_losses
from src.metrics import get_metrics
from src.models import CRNN
from src.utils import load_object

AVAILABLE_MODULES = dict()


def register_module(name):
    def decorator(f):
        global AVAILABLE_MODULES
        AVAILABLE_MODULES[name] = f
        return f

    return decorator


@register_module("classification")
class ClassificationModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self._config = config

        self._model = create_model(
            num_classes=self._config.num_classes, **self._config.model_kwargs
        )
        self._losses = get_losses(self._config.losses)
        metrics = get_metrics(
            experiment=self._config.experiment,
            num_classes=self._config.num_classes,
            num_labels=self._config.num_classes,
            task="multilabel",
            average="weighted",
            threshold=0.5,
        )
        self._valid_metrics = metrics.clone(prefix="val_")
        self._test_metrics = metrics.clone(prefix="test_")

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def configure_optimizers(self):
        optimizer = load_object(self._config.optimizer)(
            self._model.parameters(),
            **self._config.optimizer_kwargs,
        )
        scheduler = load_object(self._config.scheduler)(optimizer, **self._config.scheduler_kwargs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self._config.monitor_metric,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        images, gt_labels = batch
        pr_logits = self(images)
        return self._calculate_loss(pr_logits, gt_labels, "train_")

    def validation_step(self, batch, batch_idx):
        images, gt_labels = batch
        pr_logits = self(images)
        self._calculate_loss(pr_logits, gt_labels, "val_")
        pr_labels = torch.sigmoid(pr_logits)
        self._valid_metrics(pr_labels, gt_labels)

    def test_step(self, batch, batch_idx):
        images, gt_labels = batch
        pr_logits = self(images)
        pr_labels = torch.sigmoid(pr_logits)
        self._test_metrics(pr_labels, gt_labels)

    def on_validation_epoch_start(self) -> None:
        self._valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self._valid_metrics.compute(), on_epoch=True)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self._test_metrics.compute(), on_epoch=True)

    def _calculate_loss(
        self,
        pr_logits: torch.Tensor,
        gt_labels: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        total_loss = 0
        for cur_loss in self._losses:
            loss = cur_loss.loss(pr_logits, gt_labels)
            total_loss += cur_loss.weight * loss
            self.log(f"{prefix}{cur_loss.name}_loss", loss.item())
        self.log(f"{prefix}total_loss", total_loss.item())
        return total_loss


@register_module("segmentation")
class SegmentationModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self._config = config

        # Create model for segmentation
        self._model = Unet(
            encoder_name=self._config.model_kwargs["model_name"],  # Backbone from timm
            encoder_weights=self._config.model_kwargs[
                "pretrained"
            ],  # Use ImageNet pre-trained weights
            classes=self._config.num_classes,  # Number of classes in the segmentation task
            activation=self._config.model_kwargs["activation"],  # Choose the activation function
        )

        # Load loss functions for segmentation
        self._losses = get_losses(self._config.losses)

        # Load metrics for segmentation
        metrics = get_metrics(experiment=self._config.experiment)
        self._valid_metrics = metrics.clone(prefix="val_")
        self._test_metrics = metrics.clone(prefix="test_")

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self._model(x))

    def configure_optimizers(self):
        optimizer = load_object(self._config.optimizer)(
            self._model.parameters(),
            **self._config.optimizer_kwargs,
        )
        scheduler = load_object(self._config.scheduler)(optimizer, **self._config.scheduler_kwargs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self._config.monitor_metric,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        images, gt_masks = batch
        pr_masks = self(images)
        return self._calculate_loss(pr_masks, gt_masks, "train_")

    def validation_step(self, batch, batch_idx):
        images, gt_masks = batch
        pr_masks = self(images)

        self._calculate_loss(pr_masks, gt_masks, "val_")

        pr_labels = (pr_masks > 0.5).int()
        self._valid_metrics(pr_labels, gt_masks)

    def test_step(self, batch, batch_idx):
        images, gt_masks = batch
        pr_masks = self(images)

        pr_labels = (pr_masks > 0.5).int()
        self._test_metrics(pr_labels, gt_masks)

    def on_validation_epoch_start(self) -> None:
        self._valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self._valid_metrics.compute(), on_epoch=True)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self._test_metrics.compute(), on_epoch=True)

    def _calculate_loss(
        self,
        pr_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        total_loss = 0
        for cur_loss in self._losses:
            loss = cur_loss.loss(pr_masks, gt_masks)
            total_loss += cur_loss.weight * loss
            self.log(f"{prefix}{cur_loss.name}_loss", loss.item())
        self.log(f"{prefix}total_loss", total_loss.item())
        return total_loss


@register_module("ocr")
class OCRModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self._config = config

        self._model = CRNN(**self._config.model_kwargs)
        self._losses = get_losses(self._config.losses)

        # Load metrics for segmentation
        metrics = get_metrics(experiment=self._config.experiment)
        self._valid_metrics = metrics.clone(prefix="val_")
        self._test_metrics = metrics.clone(prefix="test_")

        self.save_hyperparameters()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._model(tensor)

    def configure_optimizers(self):
        optimizer = load_object(self._config.optimizer)(
            self._model.parameters(),
            **self._config.optimizer_kwargs,
        )
        scheduler = load_object(self._config.scheduler)(optimizer, **self._config.scheduler_kwargs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self._config.monitor_metric,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        images, targets, target_lengths = batch
        log_probs = self(images)
        input_lengths = torch.IntTensor([log_probs.size(0)] * images.size(0))

        return self._calculate_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            "train_",
        )

    def validation_step(self, batch, batch_idx):
        images, targets, target_lengths = batch
        log_probs = self(images)
        input_lengths = torch.IntTensor([log_probs.size(0)] * images.size(0))
        self._calculate_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            "valid_",
        )
        self._valid_metrics(log_probs, targets)

    def test_step(self, batch, batch_idx):
        images, targets, target_lengths = batch
        log_probs = self(images)
        input_lengths = torch.IntTensor([log_probs.size(0)] * images.size(0))
        self._calculate_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            "valid_",
        )
        self._test_metrics(log_probs, targets)

    def on_validation_epoch_start(self) -> None:
        self._valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self._valid_metrics.compute(), on_epoch=True)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self._test_metrics.compute(), on_epoch=True)

    def _calculate_loss(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=log_probs.device)
        for cur_loss in self._losses:
            loss = cur_loss.loss(
                log_probs=log_probs,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
            )
            total_loss += cur_loss.weight * loss
            self.log(f"{prefix}{cur_loss.name}_loss", loss.item())
        self.log(f"{prefix}total_loss", total_loss.item())
        return total_loss
