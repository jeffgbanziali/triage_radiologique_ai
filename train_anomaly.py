"""
Entrainement de l'AE/VAE pour la detection d'anomalies.
L'AE est entraine sur les images normales uniquement (labels all-zero).

Usage :
  python train_anomaly.py
  python train_anomaly.py --model vae_conv
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{ROOT / 'mlflow.db'}"

import numpy as np
import torch
import mlflow

from src.setup_check import merge_configs, set_global_seeds
from src.data.dataset import ChestMNISTDataset
from src.data.augmentation import get_anomaly_transforms
from src.anomaly.autoencoder import build_autoencoder
from src.anomaly.scoring import anomaly_summary
from src.training.trainer import AnomalyTrainer, build_optimizer
from src.training.mlflow_utils import start_run, log_config, log_model, log_figure, mark_best_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ae_conv", choices=["ae_conv", "vae_conv"])
    args = parser.parse_args()

    cfg = merge_configs("base", "anomaly")
    cfg["model"] = args.model
    set_global_seeds(cfg["seed"])

    resolution  = cfg["resolution"]
    batch_size  = cfg["batch_size"]
    num_workers = cfg["num_workers"]
    device      = cfg["device"]
    data_dir    = cfg["raw_dir"] + "chestmnist"
    is_vae      = args.model == "vae_conv"

    # Donnees
    transform = get_anomaly_transforms(resolution)
    train_full = ChestMNISTDataset("train", resolution, transform, data_dir)
    val_full   = ChestMNISTDataset("val",   resolution, transform, data_dir)
    test_full  = ChestMNISTDataset("test",  resolution, transform, data_dir)

    train_normal_idx = train_full.get_normal_indices()
    val_normal_idx   = val_full.get_normal_indices()
    print(f"Normaux — train: {len(train_normal_idx)} | val: {len(val_normal_idx)}")
    print(f"Total   — train: {len(train_full)}       | test: {len(test_full)}")

    normal_train_ds = ChestMNISTDataset("train", resolution, transform, data_dir, train_normal_idx)
    normal_val_ds   = ChestMNISTDataset("val",   resolution, transform, data_dir, val_normal_idx)

    train_loader = torch.utils.data.DataLoader(
        normal_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        normal_val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    # Test loader : toutes les images (normaux + pathologiques) pour évaluer le score
    test_loader = torch.utils.data.DataLoader(
        test_full, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Modele
    model = build_autoencoder(cfg)
    optimizer = build_optimizer(model, cfg)

    exp_name = f"{cfg['mlflow_experiment_prefix']}-anomaly"
    run_name = f"{args.model.upper()}_anomaly"

    with start_run(exp_name, run_name) as run:
        log_config(cfg)
        mlflow.set_tag("model_type", args.model)
        mlflow.log_param("n_normal_train", len(train_normal_idx))

        trainer = AnomalyTrainer(model, optimizer, device, is_vae)
        trainer.fit(
            train_loader, val_loader,
            max_epochs=cfg.get("max_epochs", 30),
            patience=cfg.get("patience", 7),
            save_path=f"data/processed/anomaly_{args.model}_best.pt",
        )

        # Score d'anomalie et seuil
        summary = anomaly_summary(
            model, val_loader, test_loader,
            device=device,
            percentile=cfg.get("anomaly_threshold_percentile", 95),
            is_vae=is_vae,
        )
        threshold = summary["threshold"]
        anomaly_rate = summary["anomaly_rate"]

        mlflow.log_metric("anomaly_threshold", threshold)
        mlflow.log_metric("test_anomaly_rate", float(anomaly_rate))

        # Visualisation : exemples de reconstructions
        fig = _plot_reconstructions(model, test_loader, device, is_vae)
        log_figure(fig, f"reconstructions_{args.model}.png")
        import matplotlib.pyplot as plt; plt.close(fig)

        # Histogramme des scores d'anomalie
        fig2 = _plot_score_histogram(
            summary["normal_scores"], summary["test_scores"], threshold
        )
        log_figure(fig2, f"anomaly_score_histogram_{args.model}.png")
        plt.close(fig2)

        log_model(model, f"model_{args.model}")
        run_id = run.info.run_id
        mark_best_run(run_id, "anomaly", artifact_key=f"model_{args.model}")

        print(f"\nRun ID MLflow : {run_id}")
        print(f"Seuil d'anomalie : {threshold:.5f}")
        print(f"Taux d'anomalie (test) : {100*anomaly_rate:.1f}%")


def _plot_reconstructions(model, loader, device, is_vae, n=8):
    """Génère une grille image originale / reconstruction."""
    import matplotlib.pyplot as plt
    model.eval()
    imgs, _ = next(iter(loader))
    imgs = imgs[:n].to(device)
    with torch.no_grad():
        if is_vae:
            x_hat, _, _ = model(imgs)
        else:
            x_hat, _ = model(imgs)

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        img_np = imgs[i].cpu().permute(1, 2, 0).numpy()
        rec_np = x_hat[i].cpu().permute(1, 2, 0).numpy()
        img_np = img_np.clip(0, 1)
        rec_np = rec_np.clip(0, 1)
        axes[0, i].imshow(img_np, cmap="gray" if img_np.shape[2] == 1 else None)
        axes[1, i].imshow(rec_np, cmap="gray" if rec_np.shape[2] == 1 else None)
        axes[0, i].axis("off"); axes[1, i].axis("off")
    axes[0, 0].set_title("Original", loc="left")
    axes[1, 0].set_title("Reconstruit", loc="left")
    fig.suptitle("Exemples de reconstructions AE/VAE")
    return fig


def _plot_score_histogram(normal_scores, test_scores, threshold):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(normal_scores, bins=50, alpha=0.6, label="Normaux (val)", color="green")
    ax.hist(test_scores,   bins=50, alpha=0.6, label="Test (tous)",   color="orange")
    ax.axvline(threshold, color="red", linestyle="--", label=f"Seuil={threshold:.4f}")
    ax.set_xlabel("Score d'anomalie (MSE reconstruction)")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution des scores d'anomalie")
    ax.legend()
    return fig


if __name__ == "__main__":
    main()
