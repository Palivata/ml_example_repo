import torch
from timm import create_model
from torch import nn


class CRNN(nn.Module):
    """Реализует CRNN модель для OCR задачи.
    CNN-backbone берется из timm, в RNN части стоит GRU.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        cnn_output_featuers: int = 40,
        rnn_features_num: int = 48,
        rnn_hidden_size: int = 64,
        rnn_dropout: float = 0.1,
        rnn_bidirectional: bool = True,
        rnn_num_layers: int = 2,
        num_classes: int = 11,
    ) -> None:
        super().__init__()

        self.backbone = create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2,),
        )

        self.gate = nn.Conv2d(cnn_output_featuers, rnn_features_num, kernel_size=1, bias=False)

        self.rnn = nn.GRU(
            input_size=576,
            hidden_size=rnn_hidden_size,
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional,
            num_layers=rnn_num_layers,
        )

        classifier_in_features = rnn_hidden_size
        if rnn_bidirectional:
            classifier_in_features = 2 * rnn_hidden_size

        self.fc = nn.Linear(classifier_in_features, num_classes)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        cnn_features = self.backbone(tensor)[0]
        cnn_features = self.gate(cnn_features)
        cnn_features = cnn_features.permute(3, 0, 2, 1)
        cnn_features = cnn_features.reshape(
            cnn_features.shape[0],
            cnn_features.shape[1],
            cnn_features.shape[2] * cnn_features.shape[3],
        )
        rnn_output, _ = self.rnn(cnn_features)
        logits = self.fc(rnn_output)
        return self.softmax(logits)
