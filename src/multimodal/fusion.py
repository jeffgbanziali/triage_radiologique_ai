"""Strategies de fusion multimodale image + texte : late fusion et early fusion."""
import torch
import torch.nn as nn
from typing import Optional


class LateFusionClassifier(nn.Module):
    """Fusion tardive : concat logits image + texte, couche de fusion finale. Gere les textes absents (zeroed out)."""

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        num_classes: int = 14,
        img_feature_dim: int = 14,    # logits de l'encodeur image
        txt_feature_dim: int = 14,    # logits de l'encodeur texte
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder  = text_encoder

        self.fusion_head = nn.Sequential(
            nn.Linear(img_feature_dim + txt_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        images: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        img_logits = self.image_encoder(images)

        if text_mask is not None:
            text_features = text_features * text_mask.float().unsqueeze(-1)

        txt_logits = self.text_encoder(text_features) if hasattr(self.text_encoder, 'mlp') \
                     else text_features

        combined = torch.cat([img_logits, txt_logits], dim=-1)  # (B, 2*num_classes)
        return self.fusion_head(combined)


class EarlyFusionClassifier(nn.Module):
    """Fusion precoce : concatene les embeddings image et texte avant la tete de classification."""

    def __init__(
        self,
        img_embed_dim: int = 256,
        txt_embed_dim: int = 256,
        num_classes: int = 14,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(img_embed_dim + txt_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        img_embed: torch.Tensor,
        txt_embed: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if text_mask is not None:
            txt_embed = txt_embed * text_mask.float().unsqueeze(-1)
        combined = torch.cat([img_embed, txt_embed], dim=-1)
        return self.classifier(combined)


class MultimodalSystem(nn.Module):
    """Systeme complet image+texte avec 3 modes : image_only, text_only, fusion."""

    def __init__(
        self,
        image_backbone: nn.Module,
        text_encoder: nn.Module,
        img_out_dim: int,
        txt_out_dim: int,
        num_classes: int = 14,
        fusion_strategy: str = "late",
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.image_backbone   = image_backbone
        self.text_encoder     = text_encoder
        self.fusion_strategy  = fusion_strategy

        self.image_head = nn.Linear(img_out_dim, num_classes)
        self.text_head  = nn.Linear(txt_out_dim, num_classes)

        self.fusion_head = nn.Sequential(
            nn.Linear(num_classes * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        images: torch.Tensor,
        tfidf_vectors: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        mode: str = "fusion",
    ) -> torch.Tensor:
        img_feat = self.image_backbone(images)
        txt_feat = self.text_encoder(tfidf_vectors)

        if mode == "image_only":
            return self.image_head(img_feat)

        if mode == "text_only":
            return self.text_head(txt_feat)

        img_logits = self.image_head(img_feat)
        txt_logits = self.text_head(txt_feat)
        if text_mask is not None:
            txt_logits = txt_logits * text_mask.float().unsqueeze(-1)
        combined = torch.cat([img_logits, txt_logits], dim=-1)
        return self.fusion_head(combined)
