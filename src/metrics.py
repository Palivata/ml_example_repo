import itertools

import numpy as np
import torch
from nltk.metrics.distance import edit_distance as ed
from torchmetrics import Dice, F1Score, Metric, MetricCollection, Precision, Recall
from torchmetrics.classification import BinaryJaccardIndex


class StringMatchMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state(
            "correct",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        batch_size = torch.tensor(target.shape[0])

        metric = torch.tensor(string_match(preds, target))

        self.correct += metric * batch_size
        self.total += batch_size

    def compute(self):
        return self.correct / self.total


class EditDistanceMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state(
            "correct",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        batch_size = torch.tensor(target.shape[0])

        metric = torch.tensor(edit_distance(preds, target))

        self.correct += metric * batch_size
        self.total += batch_size

    def compute(self):
        return self.correct / self.total


def string_match(pred: torch.Tensor, true: torch.Tensor) -> float:
    pred = pred.permute(1, 0, 2)
    pred = torch.Tensor.argmax(pred, dim=2).detach().cpu().numpy()

    true = true.detach().cpu().numpy()

    valid = 0
    for j in range(pred.shape[0]):
        p3 = [k for k, g in itertools.groupby(pred[j])]
        p3 = [k for k in p3 if k > 0]
        t = [k for k in true[j] if k > 0]
        valid += float(np.array_equal(p3, t))

    return valid / pred.shape[0]


def edit_distance(pred: torch.Tensor, true: torch.Tensor) -> float:
    pred = pred.permute(1, 0, 2)
    pred = torch.Tensor.argmax(pred, dim=2).detach().cpu().numpy()

    true = true.detach().cpu().numpy()

    dist = 0
    for j in range(pred.shape[0]):
        p3 = [k for k, g in itertools.groupby(pred[j])]
        p3 = [k for k in p3 if k > 0]
        t = [k for k in true[j] if k > 0]

        s_pred = "".join(list(map(lambda x: chr(x), p3)))
        s_true = "".join(list(map(lambda x: chr(x), t)))

        dist += ed(s_pred, s_true)

    return dist / pred.shape[0]


def get_metrics(experiment="classification", **kwargs) -> MetricCollection:
    if experiment == "classification":
        return MetricCollection(
            {"f1": F1Score(**kwargs), "precision": Precision(**kwargs), "recall": Recall(**kwargs)}
        )
    elif experiment == "segmentation":
        return MetricCollection({"iou": BinaryJaccardIndex(), "dice": Dice()})
    elif experiment == "ocr":
        return MetricCollection(
            {
                "string_match": StringMatchMetric(),
                "edit_distance": EditDistanceMetric(),
            }
        )
    raise ValueError(f"Unknown experiment {experiment}")
