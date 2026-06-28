"""CNN entraine depuis zero pour la classification multi-label thoracique (14 classes)."""
import torch
import torch.nn as nn
from typing import List


class ConvBlock(nn.Module):
    """Bloc Conv-BN-ReLU-MaxPool."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNFromScratch(nn.Module):
    """CNN 4 blocs pour classification multi-label thoracique (14 classes), resolution 64x64."""

    def __init__(
        self,
        num_classes: int = 14,
        base_channels: int = 32,
        num_blocks: int = 4,
        dropout: float = 0.5,
        in_channels: int = 3,
    ):
        super().__init__()
        channels: List[int] = [in_channels] + [
            base_channels * (2 ** i) for i in range(num_blocks)
        ]  # [3, 32, 64, 128, 256]

        self.features = nn.Sequential(
            *[ConvBlock(channels[i], channels[i + 1]) for i in range(num_blocks)]
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        fc_in = channels[-1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_in, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def build_cnn_from_scratch(cfg: dict) -> CNNFromScratch:
    return CNNFromScratch(
        num_classes=cfg.get("num_classes", 14),
        base_channels=cfg.get("base_channels", 32),
        num_blocks=cfg.get("num_conv_blocks", 4),
        dropout=cfg.get("dropout", 0.5),
    )
