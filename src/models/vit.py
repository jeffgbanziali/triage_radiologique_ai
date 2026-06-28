"""
DeiT-tiny (Vision Transformer) fine-tune pour la classification multi-label thoracique.
16 patchs 16x16 sur image 64x64, interpolation des positional embeddings via timm.
"""
import torch
import torch.nn as nn


class DeiTTinyClassifier(nn.Module):
    """DeiT-tiny adapte pour la classification multi-label thoracique (14 classes)."""

    def __init__(
        self,
        num_classes: int = 14,
        img_size: int = 64,
        patch_size: int = 16,
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("Installez timm : pip install timm")

        self.backbone = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,
        )

        embed_dim = self.backbone.num_features  # 192 pour DeiT-tiny

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = self.backbone(x)
        return self.head(cls_token)

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_vit(cfg: dict) -> DeiTTinyClassifier:
    return DeiTTinyClassifier(
        num_classes=cfg.get("num_classes", 14),
        img_size=cfg.get("img_size", 64),
        patch_size=cfg.get("patch_size", 16),
        pretrained=cfg.get("pretrained", True),
        dropout=cfg.get("dropout", 0.1),
    )
