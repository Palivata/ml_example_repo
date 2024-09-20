# Repository for tasks in the following areas of CV:
## - Classification
## - Binary Segmentation
## - OCR

## Prerequisites

The machine should have the following installed: Docker, Docker Compose, Make, Python 3.10, ClearML.

For successful logging, you will need the clearml.conf file, which can be obtained by running ```clearml-init``` and adding the configuration dictionary from https://app.clear.ml. Then place the clearml.conf file in the project repository.


## Config structure
The configuration for training is located [here](configs/config.yaml)
```yaml
project_name: "str: Clearml project name"
experiment_name: "str: clearml experiment name"
experiment: "str: ocr, segmentation, classification"
num_classes: "int: number of output classes"
n_epochs: "int: number of epoch to train"
accelerator: "str: gpu/cpu"
device: "int: device to train if gpu"
monitor_metric: "str: metric to choose best model"
monitor_mode: "str: max/min"
data_path: "str: path to folder with dataset"
batch_size: "int: batch_size"
num_iterations: "int: number of steps in 1 epoch"
n_workers: "int: CPU workers"
train_size: "float: dataset split to x train, (1-x)/2 val, (1-x/2) test"
width: &width "int: width of input shape"
height: &height "int: height of input shape"
seed: "int: global seed"

model_kwargs:
  model_name: "str: timm model name"
  pretrained: true

optimizer: 'torch.optim.AdamW'
optimizer_kwargs:
  lr: 1e-4
  weight_decay: 1e-5

scheduler: 'torch.optim.lr_scheduler.CosineAnnealingLR'
scheduler_kwargs:
  T_max: 10
  eta_min: 1e-5

losses:
  - name: 'bce'
    weight: 1.0
    loss_fn: 'torch.nn.BCEWithLogitsLoss'
    loss_kwargs: {}
```

A bit about augmentations:
The full list of available augmentations is located [here](src/augmentations/augmentations.py).
 To add a new augmentation, add a class similar to ```BaseAugmentation```.
In the config, the full list of augmentations is specified like this:
```
augmentations:
  train -> where to apply:
    Compose -> apply one by one or OneOf - apply only one of augmentation:
      color_jitter -> name of augmentation:
        p: 0.1 -> probability
    ...
      random_scaling:
        p: 0.1
        angle_low: 0
        angle_max: 45
        scale_low: 1
        scale_max: 1.3
      random_gaussian_blur:
        p: 0.1
        kernel_low: 7
        kernel_max: 15
      discrete_flip_rotate:
        p: 1
      resize:
        p: 1
        height: *height
        width: *width
        resize_method: bicubic
  test:
    Compose:
      resize:
        p: 1
        height: *height
        width: *width
        resize_method: bicubic
```

All fields are validated for each augmentation class.
## Development

All paths are located in the Makefile:
```
make initial
```
You can develop and modify the code.

## Training
Modify the necessary fields in the [config](configs/config.yaml):
```
make install
make train
```
## Convertation model

```
make convert_to_ov
```
