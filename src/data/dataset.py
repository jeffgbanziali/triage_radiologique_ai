"""Dataset PyTorch pour ChestMNIST : 112 120 radiographies, 14 pathologies binaires (BCEWithLogitsLoss)."""
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
NUM_CLASSES = len(LABEL_NAMES)


class ChestMNISTDataset(Dataset):
    """Wrapper PyTorch autour de medmnist.ChestMNIST avec conversion RGB et labels float32."""

    def __init__(
        self,
        split: str,
        resolution: int = 64,
        transform: Optional[object] = None,
        data_dir: str = "data/raw",
        indices: Optional[np.ndarray] = None,
    ):
        try:
            import medmnist
            from medmnist import ChestMNIST
        except ImportError as e:
            raise ImportError("Installez medmnist : pip install medmnist") from e

        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        ds = ChestMNIST(
            split=split,
            size=resolution,
            download=True,
            root=str(data_dir),
            as_rgb=True,   # grayscale -> RGB (3 canaux identiques)
        )

        self.imgs: np.ndarray = ds.imgs          # shape (N, H, W, 3), uint8
        self.labels: np.ndarray = ds.labels      # shape (N, 14), int {0,1}

        if indices is not None:
            self.imgs = self.imgs[indices]
            self.labels = self.labels[indices]

        self.transform = transform
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.imgs[idx]          # (H, W, 3) uint8
        label = self.labels[idx]      # (14,) int

        # PIL -> transforms pipeline
        from PIL import Image
        img = Image.fromarray(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # float32 requis par BCEWithLogitsLoss
        label = torch.tensor(label, dtype=torch.float32)
        return img, label

    @property
    def label_names(self):
        return LABEL_NAMES

    def get_pos_weight(self) -> torch.Tensor:
        """Calcule pos_weight[c] = n_negatifs / n_positifs pour BCEWithLogitsLoss."""
        n_pos = self.labels.sum(axis=0).astype(float)          # (14,)
        n_neg = len(self.labels) - n_pos
        pos_weight = np.where(n_pos > 0, n_neg / np.maximum(n_pos, 1), 1.0)
        return torch.tensor(pos_weight, dtype=torch.float32)

    def get_normal_indices(self) -> np.ndarray:
        """Retourne les indices des images sans pathologie (labels all-zero)."""
        return np.where(self.labels.sum(axis=1) == 0)[0]
