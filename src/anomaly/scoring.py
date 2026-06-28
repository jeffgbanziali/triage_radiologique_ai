"""Score d'anomalie (MSE reconstruction) et calibration du seuil pour l'AE/VAE."""
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_reconstruction_scores(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    is_vae: bool = False,
) -> np.ndarray:
    """Calcule le score d'anomalie MSE pour chaque image. Retourne np.ndarray (N,)."""
    model.eval()
    model.to(device)
    scores = []

    with torch.no_grad():
        for batch in dataloader:
            # dataloader peut retourner (imgs, labels) ou juste (imgs,)
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = imgs.to(device)

            if is_vae:
                x_hat, mu, logvar = model(imgs)
            else:
                x_hat, _ = model(imgs)

            # MSE pixel-wise, moyenne sur H*W*C -> 1 scalaire par image
            mse = ((imgs - x_hat) ** 2).mean(dim=(1, 2, 3))  # (B,)
            scores.append(mse.cpu().numpy())

    return np.concatenate(scores)


def calibrate_threshold(
    normal_scores: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """Seuil = percentile p des scores normaux (val set). p=95 cible un FPR d'environ 5%."""
    threshold = float(np.percentile(normal_scores, percentile))
    print(f"Seuil anomalie (percentile {percentile:.0f}%) : {threshold:.6f}")
    return threshold


def predict_anomaly(
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Retourne un tableau bool : True = atypique (score > seuil)."""
    return scores > threshold


def anomaly_summary(
    model: nn.Module,
    normal_loader: DataLoader,
    test_loader: DataLoader,
    device: str = "cpu",
    percentile: float = 95.0,
    is_vae: bool = False,
) -> Dict[str, object]:
    """Calcule les scores val (calibration seuil) et test, retourne un resume dict."""
    print("Calcul des scores sur le val set normal (calibration seuil)...")
    normal_scores = compute_reconstruction_scores(model, normal_loader, device, is_vae)
    threshold = calibrate_threshold(normal_scores, percentile)

    print("Calcul des scores sur le test set...")
    test_scores = compute_reconstruction_scores(model, test_loader, device, is_vae)
    predictions = predict_anomaly(test_scores, threshold)

    return {
        "threshold": threshold,
        "normal_scores": normal_scores,
        "test_scores": test_scores,
        "predictions": predictions,
        "anomaly_rate": predictions.mean(),
    }
