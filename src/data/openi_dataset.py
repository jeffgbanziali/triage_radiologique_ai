"""
Dataset multimodal OpenI - Indiana University Chest X-Rays (version Kaggle CSV).

Source Kaggle : raddar/chest-xrays-indiana-university
Structure attendue dans data/ :
  data/indiana_reports.csv       : uid, MeSH, Problems, findings, impression, ...
  data/indiana_projections.csv   : uid, filename, projection
  data/images/images_normalized/ : 7470 images PNG
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


OPENI_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

# Mapping mots-cles du champ Problems vers index de label
_KEYWORD_MAP = {
    "atelectasis": 0,
    "cardiomegaly": 1,
    "effusion": 2,
    "infiltrat": 3,      # infiltrate, infiltration
    "mass": 4,
    "nodule": 5,
    "pneumonia": 6,
    "pneumothorax": 7,
    "consolidation": 8,
    "edema": 9,
    "emphysema": 10,
    "fibrosis": 11,
    "pleural": 12,       # pleural thickening, pleural effusion (si absent dans effusion)
    "hernia": 13,
}


def _problems_to_labels(problems_str: str) -> np.ndarray:
    """Convertit la colonne Problems en vecteur binaire 14 classes."""
    vec = np.zeros(14, dtype=np.float32)
    if not isinstance(problems_str, str) or problems_str.lower() == "normal":
        return vec
    text_lower = problems_str.lower()
    for keyword, idx in _KEYWORD_MAP.items():
        if keyword in text_lower:
            vec[idx] = 1.0
    return vec


def build_openi_dataframe(data_dir: str) -> pd.DataFrame:
    """
    Construit un DataFrame avec colonnes : uid, image_path, text, labels.
    Utilise les CSV du dataset Kaggle (indiana_reports.csv + indiana_projections.csv).

    Args:
        data_dir: chemin vers le dossier data/ (contient les CSV et data/images/).
    """
    data_dir = Path(data_dir)
    reports_csv     = data_dir / "indiana_reports.csv"
    projections_csv = data_dir / "indiana_projections.csv"
    images_dir      = data_dir / "images" / "images_normalized"

    if not reports_csv.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {reports_csv}\n"
            "Telechargez le dataset Kaggle 'raddar/chest-xrays-indiana-university' "
            "et extrayez-le dans data/."
        )

    reports     = pd.read_csv(reports_csv)
    projections = pd.read_csv(projections_csv)

    # Garder uniquement les vues frontales (plus informatives en radiologie thoracique)
    frontal = projections[projections["projection"] == "Frontal"].copy()

    # Joindre rapports + images frontales sur uid
    merged = frontal.merge(reports[["uid", "findings", "impression", "Problems"]], on="uid", how="inner")

    def build_text(row):
        parts = []
        if isinstance(row["findings"], str) and row["findings"].strip():
            parts.append(row["findings"].strip())
        if isinstance(row["impression"], str) and row["impression"].strip():
            parts.append(row["impression"].strip())
        return " ".join(parts)

    merged["text"]       = merged.apply(build_text, axis=1)
    merged["labels"]     = merged["Problems"].apply(_problems_to_labels)
    merged["image_path"] = merged["filename"].apply(lambda fn: str(images_dir / fn))

    merged = merged[merged["image_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)

    print(f"OpenI CSV : {len(merged)} paires image-rapport valides (vues frontales).")
    print(f"  Prevalence labels : {merged['labels'].apply(pd.Series).mean().round(3).to_dict()}")
    return merged[["uid", "image_path", "text", "labels"]]


class OpenIDataset(Dataset):
    """
    Dataset multimodal PyTorch pour OpenI.
    Retourne (image_tensor, text_str, label_tensor).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_transform=None,
        resolution: int = 64,
        indices: Optional[np.ndarray] = None,
    ):
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        self.df = df
        self.resolution = resolution
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, torch.Tensor]:
        row = self.df.iloc[idx]

        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.resolution, self.resolution), 0)

        img_tensor = self.image_transform(img)
        text       = str(row.get("text", "") or "")
        labels     = torch.tensor(row["labels"], dtype=torch.float32)

        return img_tensor, text, labels
