"""
Repair script: log saved DenseNet121 weights into MLflow.
Run from project root: python scripts/repair_densenet_mlflow.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # mark_best_run uses relative path configs/base.yaml
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
import torch.nn as nn
import mlflow

from src.setup_check import merge_configs
from src.models.transfer_learning import build_densenet121
from src.data.dataset import ChestMNISTDataset
from src.data.augmentation import get_val_transforms
from src.evaluation.metrics import compute_metrics, print_metrics, plot_confusion_matrices
from src.training.trainer import Trainer
from src.training.mlflow_utils import (
    start_run, log_config, log_model, log_figure, mark_best_run,
    MLFLOW_DB_URI,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mlflow.set_tracking_uri(MLFLOW_DB_URI)

cfg = merge_configs("base", "transfer")
device = cfg["device"]
resolution = cfg["resolution"]
model_path = ROOT / "data" / "processed" / "densenet_finetuned.pt"

if not model_path.exists():
    print(f"ERREUR : modele introuvable -> {model_path}")
    sys.exit(1)

print(f"Chargement DenseNet121 depuis {model_path.name}...")
model = build_densenet121(cfg)
model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
model.eval()

print("Chargement test set ChestMNIST...")
transform = get_val_transforms(resolution)
test_ds = ChestMNISTDataset(
    "test", resolution, transform,
    data_dir=str(ROOT / "data" / "raw" / "chestmnist"),
)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0,
)

# Simple BCE without pos_weight for inference only
criterion = nn.BCEWithLogitsLoss()
trainer = Trainer(model, None, criterion, None, device, cfg)

print(f"Inference sur {len(test_ds)} images (CPU)...")
val_loss, probs, labels = trainer.val_epoch(test_loader)
metrics = compute_metrics(labels.numpy(), probs.numpy())
print_metrics(metrics)

exp_name = f"{cfg['mlflow_experiment_prefix']}-classification"
print("Logging vers MLflow...")
with start_run(exp_name, "DenseNet121_transfer_repaired") as run:
    log_config(cfg)
    mlflow.set_tag("model_type", "densenet")
    mlflow.set_tag("phase", "fine-tuned")
    mlflow.set_tag("repaired", "true")
    mlflow.log_param("model_file", "densenet_finetuned.pt")
    mlflow.log_param("phase1_best_epoch", 4)
    mlflow.log_param("phase2_epochs", 3)
    mlflow.log_metrics({
        "test_val_loss": val_loss,
        **{f"test_{k}": v for k, v in metrics.items() if isinstance(v, float)},
    })

    fig = plot_confusion_matrices(labels.numpy(), probs.numpy())
    log_figure(fig, "confusion_matrix_densenet.png")
    plt.close(fig)

    log_model(model, artifact_name="model_densenet")
    run_id = run.info.run_id
    mark_best_run(run_id, "classification", artifact_key="model_densenet")

print(f"\nRun ID : {run_id}")
print(f"AUC macro : {metrics['auc_macro']:.4f}")
print(f"F1  macro : {metrics['f1_macro']:.4f}")
print("DenseNet MLflow repair -> OK")
