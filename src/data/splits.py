"""Splits train/val/test stratifies multi-label via iterative-stratification."""
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def make_multilabel_splits(
    labels: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    save_dir: str = "data/splits",
) -> Dict[str, np.ndarray]:
    """Cree un split train/val/test stratifie multi-label via MultilabelStratifiedShuffleSplit."""
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError:
        raise ImportError(
            "Installez iterative-stratification : pip install iterative-stratification"
        )

    N = len(labels)
    all_idx = np.arange(N)

    msss_test = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    trainval_idx, test_idx = next(msss_test.split(all_idx.reshape(-1, 1), labels))

    val_fraction_of_trainval = val_size / (1.0 - test_size)
    msss_val = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction_of_trainval, random_state=seed
    )
    train_sub_idx, val_sub_idx = next(
        msss_val.split(
            trainval_idx.reshape(-1, 1),
            labels[trainval_idx],
        )
    )
    train_idx = trainval_idx[train_sub_idx]
    val_idx   = trainval_idx[val_sub_idx]

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for split_name, idx in splits.items():
        out_path = Path(save_dir) / f"{split_name}_indices.json"
        with open(out_path, "w") as f:
            json.dump(idx.tolist(), f)

    _print_split_stats(labels, splits)
    return splits


def load_splits(save_dir: str = "data/splits") -> Dict[str, np.ndarray]:
    """Recharge les indices sauvegardes depuis le dossier splits/."""
    splits = {}
    for split_name in ("train", "val", "test"):
        path = Path(save_dir) / f"{split_name}_indices.json"
        with open(path, "r") as f:
            splits[split_name] = np.array(json.load(f))
    return splits


def _print_split_stats(labels: np.ndarray, splits: Dict[str, np.ndarray]) -> None:
    total = sum(len(v) for v in splits.values())
    print(f"\nSplit - total : {total} echantillons")
    for name, idx in splits.items():
        n = len(idx)
        prev = labels[idx].mean(axis=0)
        print(f"  {name:6s} : {n:6d} ({100*n/total:.1f}%) | "
              f"prevalence moy={100*prev.mean():.2f}%")
