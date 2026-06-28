"""
Metriques d'evaluation pour la classification multi-label : AUC, F1, MCC, Balanced Accuracy.
"""
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

LABEL_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    label_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Calcule AUC, F1, MCC, Balanced Accuracy (global + per-classe) pour un probleme multi-label."""
    label_names = label_names or LABEL_NAMES[:y_true.shape[1]]
    y_pred = (y_prob >= threshold).astype(int)
    num_classes = y_true.shape[1]

    metrics: Dict[str, float] = {}

    # Metriques globales
    metrics["f1_macro"]  = float(f1_score(y_true, y_pred, average="macro",  zero_division=0))
    metrics["f1_micro"]  = float(f1_score(y_true, y_pred, average="micro",  zero_division=0))
    metrics["f1_samples"] = float(f1_score(y_true, y_pred, average="samples", zero_division=0))

    # MCC multi-label : moyenne des MCC per-classe (extension naturelle)
    mcc_per_class = []
    for c in range(num_classes):
        if y_true[:, c].sum() > 0:  # skip classes absentes dans ce split
            mcc_per_class.append(matthews_corrcoef(y_true[:, c], y_pred[:, c]))
    metrics["mcc_macro"] = float(np.mean(mcc_per_class)) if mcc_per_class else 0.0

    # Balanced Accuracy multi-label : moyenne sur les classes
    bal_acc_per_class = []
    for c in range(num_classes):
        if y_true[:, c].sum() > 0 and (1 - y_true[:, c]).sum() > 0:
            bal_acc_per_class.append(
                balanced_accuracy_score(y_true[:, c], y_pred[:, c])
            )
    metrics["balanced_accuracy"] = float(np.mean(bal_acc_per_class)) if bal_acc_per_class else 0.0

    # AUC-ROC macro (ignore les classes sans positif)
    try:
        auc_classes = []
        for c in range(num_classes):
            if y_true[:, c].sum() > 0 and (1 - y_true[:, c]).sum() > 0:
                auc_classes.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
        metrics["auc_macro"] = float(np.mean(auc_classes)) if auc_classes else 0.0
    except Exception:
        metrics["auc_macro"] = 0.0

    # Average Precision (mAP)
    try:
        metrics["map"] = float(average_precision_score(y_true, y_prob, average="macro"))
    except Exception:
        metrics["map"] = 0.0

    # Metriques per-classe
    for c, name in enumerate(label_names):
        if y_true[:, c].sum() > 0:
            metrics[f"auc_{name}"] = float(
                roc_auc_score(y_true[:, c], y_prob[:, c])
                if (1 - y_true[:, c]).sum() > 0 else 0.0
            )
            metrics[f"f1_{name}"]  = float(
                f1_score(y_true[:, c], y_pred[:, c], zero_division=0)
            )

    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """Affiche un resume des metriques globales."""
    print("\n" + "-" * 55)
    print(f"  AUC macro       : {metrics.get('auc_macro', 0):.4f}")
    print(f"  mAP             : {metrics.get('map', 0):.4f}")
    print(f"  F1  macro       : {metrics.get('f1_macro', 0):.4f}")
    print(f"  F1  micro       : {metrics.get('f1_micro', 0):.4f}")
    print(f"  MCC macro       : {metrics.get('mcc_macro', 0):.4f}")
    print(f"  Balanced Acc    : {metrics.get('balanced_accuracy', 0):.4f}")
    print("-" * 55)


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    label_names: Optional[List[str]] = None,
    max_classes: int = 14,
):
    """Genere une grille de matrices de confusion binaires (une par classe)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    label_names = label_names or LABEL_NAMES[:y_true.shape[1]]
    y_pred = (y_prob >= threshold).astype(int)
    n = min(len(label_names), max_classes)
    cols = 7
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()

    for c in range(n):
        cm = confusion_matrix(y_true[:, c], y_pred[:, c])
        im = axes[c].imshow(cm, interpolation="nearest", cmap="Blues")
        axes[c].set_title(label_names[c], fontsize=8)
        axes[c].set_xticks([0, 1])
        axes[c].set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                axes[c].text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                             color="white" if cm[i, j] > cm.max() / 2 else "black")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Matrices de confusion binaires par classe", fontsize=12)
    plt.tight_layout()
    return fig
