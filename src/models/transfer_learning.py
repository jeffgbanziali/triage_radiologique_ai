"""
DenseNet121 pre-entraine sur ImageNet pour la classification multi-label thoracique.
Transfer learning en 2 phases : backbone gele puis fine-tuning du denseblock4.
"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class DenseNet121Classifier(nn.Module):
    """DenseNet121 avec tete de classification multi-label (14 classes) remplacant la tete ImageNet."""

    def __init__(self, num_classes: int = 14, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()

        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        self.features = backbone.features
        in_features = backbone.classifier.in_features

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),  # logits
        )

        self.freeze_backbone()

    def freeze_backbone(self):
        """Phase 1 : gele le backbone (feature extractor)."""
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_last_block(self, block_name: str = "denseblock4"):
        """Phase 2 : degele le dernier bloc dense pour le fine-tuning."""
        for name, p in self.features.named_parameters():
            if block_name in name or "norm5" in name:
                p.requires_grad = True

    def unfreeze_all(self):
        """Degele tout le backbone (fine-tuning complet, LR tres faible requis)."""
        for p in self.features.parameters():
            p.requires_grad = True

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)


def build_densenet121(cfg: dict) -> DenseNet121Classifier:
    return DenseNet121Classifier(
        num_classes=cfg.get("num_classes", 14),
        dropout=cfg.get("dropout", 0.3),
        pretrained=cfg.get("pretrained", True),
    )
