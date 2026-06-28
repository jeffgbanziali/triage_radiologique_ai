"""Encodeurs texte pour les comptes-rendus radiologiques : TF-IDF + MLP (baseline CPU)."""
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


class TFIDFMLPEncoder(nn.Module):
    """Encodeur texte TF-IDF + MLP : vectoriseur gele, seul le MLP est entrainable."""

    def __init__(
        self,
        vocab_size: int = 5000,
        output_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.output_dim = output_dim

        # Projection TF-IDF (vocab_size -> output_dim) via MLP 2 couches
        self.mlp = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        self.vectorizer = None

    def fit(self, texts: List[str]) -> "TFIDFMLPEncoder":
        """Entraine le vectoriseur TF-IDF sur le corpus."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.vocab_size,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        )
        self.vectorizer.fit(texts)
        return self

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Vectorise une liste de textes en tenseur TF-IDF (N, vocab_size)."""
        if self.vectorizer is None:
            raise RuntimeError("Appelez d'abord fit() sur le corpus.")
        mat = self.vectorizer.transform(texts).toarray().astype(np.float32)
        return torch.tensor(mat)

    def forward(self, tfidf_vectors: torch.Tensor) -> torch.Tensor:
        return self.mlp(tfidf_vectors)

    def handle_missing(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masque les embeddings des textes absents (remplace par zero)."""
        return embeddings * mask.float().unsqueeze(-1)


class BioClinicalBERTEncoder(nn.Module):
    """Encodeur BioClinicalBERT (HuggingFace, ~400MB). Extrait le [CLS] token."""

    def __init__(self, output_dim: int = 256, freeze_bert: bool = True):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("Installez transformers : pip install transformers")

        model_name = "emilyalsentzer/Bio_ClinicalBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        if freeze_bert:
            # Gele BERT : seule la tete de projection est entrainable
            for p in self.bert.parameters():
                p.requires_grad = False

        bert_dim = self.bert.config.hidden_size  # 768
        self.projection = nn.Sequential(
            nn.Linear(bert_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, texts: List[str], device: str = "cpu") -> torch.Tensor:
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.bert(**encoding)
        cls_token = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        return self.projection(cls_token)               # (B, output_dim)
