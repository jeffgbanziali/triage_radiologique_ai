"""
Phase 5 — Entraînement multimodal sur OpenI (Indiana University CXR).

Compare 3 modes :
  1. image_only  — DenseNet features + tête MLP
  2. text_only   — TF-IDF + MLP
  3. fusion      — fusion tardive (late fusion)

Usage :
  python train_multimodal.py
  python train_multimodal.py --mode image_only
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
from src.data.openi_dataset import build_openi_dataframe, OpenIDataset
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.splits import make_multilabel_splits
from src.models.transfer_learning import build_densenet121
from src.multimodal.text_encoder import TFIDFMLPEncoder
from src.multimodal.fusion import MultimodalSystem
from src.training.trainer import build_optimizer, build_scheduler, build_criterion
from src.training.mlflow_utils import start_run, log_config, log_model, mark_best_run
from src.evaluation.metrics import compute_metrics, print_metrics


class MultimodalTrainer:
    """Boucle d'entraînement pour le système multimodal."""

    def __init__(self, model, optimizer, criterion, tfidf_encoder, device):
        self.model          = model.to(device)
        self.optimizer      = optimizer
        self.criterion      = criterion
        self.tfidf_encoder  = tfidf_encoder
        self.device         = device

    def _prepare_batch(self, batch, mode):
        imgs, texts, labels = batch
        imgs   = imgs.to(self.device)
        labels = labels.to(self.device)

        # Vectorise les textes (TF-IDF)
        tfidf_vecs = self.tfidf_encoder.encode_texts(texts).to(self.device)
        # Masque les textes vides (modalité manquante)
        text_mask = torch.tensor([len(t.strip()) > 0 for t in texts]).to(self.device)
        return imgs, tfidf_vecs, text_mask, labels

    def train_epoch(self, loader, mode="fusion"):
        self.model.train()
        total_loss, n = 0.0, 0
        for batch in loader:
            imgs, tfidf_vecs, text_mask, labels = self._prepare_batch(batch, mode)
            self.optimizer.zero_grad()
            logits = self.model(imgs, tfidf_vecs, text_mask, mode=mode)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        return total_loss / n

    @torch.no_grad()
    def val_epoch(self, loader, mode="fusion"):
        self.model.eval()
        total_loss, n = 0.0, 0
        all_probs, all_labels = [], []
        for batch in loader:
            imgs, tfidf_vecs, text_mask, labels = self._prepare_batch(batch, mode)
            logits = self.model(imgs, tfidf_vecs, text_mask, mode=mode)
            loss = self.criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())
        return total_loss / n, torch.cat(all_probs), torch.cat(all_labels)


def run_experiment(mode: str, model, tfidf_encoder, train_loader, val_loader, test_loader,
                    cfg, pos_weight, exp_name):
    device = cfg["device"]
    criterion = build_criterion(cfg, pos_weight.to(device))
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, cfg.get("max_epochs", 20))

    from src.training.trainer import EarlyStopping
    early_stop = EarlyStopping(patience=cfg.get("patience", 5))

    trainer = MultimodalTrainer(model, optimizer, criterion, tfidf_encoder, device)

    run_name = f"multimodal_{mode}"
    with start_run(exp_name, run_name) as run:
        log_config(cfg)
        mlflow.set_tag("fusion_mode", mode)

        for epoch in range(1, cfg.get("max_epochs", 20) + 1):
            train_loss = trainer.train_epoch(train_loader, mode)
            val_loss, probs, labels = trainer.val_epoch(val_loader, mode)
            metrics = compute_metrics(labels.numpy(), probs.numpy())
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss,
                 "val_auc_macro": metrics["auc_macro"], "val_f1_macro": metrics["f1_macro"]},
                step=epoch,
            )
            print(f"Epoch {epoch:3d} | {mode:12s} | train={train_loss:.4f} | "
                  f"val={val_loss:.4f} | AUC={metrics['auc_macro']:.4f}")
            if scheduler:
                scheduler.step()
            if early_stop.step(val_loss, model):
                print(f"Early stopping à l'epoch {epoch}.")
                break

        early_stop.restore_best(model)

        # Évaluation finale
        _, probs_test, labels_test = trainer.val_epoch(test_loader, mode)
        test_metrics = compute_metrics(labels_test.numpy(), probs_test.numpy())
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()
                            if isinstance(v, float)})
        print_metrics(test_metrics)

        run_id = run.info.run_id
        auc = test_metrics.get("auc_macro", 0.0)
        print(f"Run ID : {run_id} | AUC test : {auc:.4f}")
    return run_id, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["image_only", "text_only", "fusion", "all"])
    args = parser.parse_args()

    cfg = merge_configs("base", "multimodal")
    set_global_seeds(cfg["seed"])
    device = cfg["device"]
    resolution = cfg["resolution"]

    # Chargement OpenI
    try:
        df = build_openi_dataframe(cfg["openi_dir"])
    except FileNotFoundError as e:
        print(f"ERREUR : {e}")
        print("Lancez d'abord : python data/download_datasets.py --dataset openi")
        sys.exit(1)

    labels_array = np.stack(df["labels"].values)
    splits = make_multilabel_splits(labels_array, seed=cfg["seed"])

    train_ds = OpenIDataset(df, get_train_transforms(resolution), resolution, splits["train"])
    val_ds   = OpenIDataset(df, get_val_transforms(resolution),   resolution, splits["val"])
    test_ds  = OpenIDataset(df, get_val_transforms(resolution),   resolution, splits["test"])

    bs = cfg["batch_size"]
    nw = cfg["num_workers"]
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, collate_fn=_collate)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, collate_fn=_collate)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw, collate_fn=_collate)

    # Encodeur texte TF-IDF
    train_texts = df.iloc[splits["train"]]["text"].fillna("").tolist()
    tfidf_enc = TFIDFMLPEncoder(
        vocab_size=cfg.get("tfidf_max_features", 5000),
        output_dim=cfg.get("num_classes", 14),
        hidden_dim=cfg.get("mlp_hidden_dim", 256),
    ).fit(train_texts)

    # Modele multimodal
    image_backbone = build_densenet121(cfg)
    # DenseNet sort 1024 features avant la tete -> on adapte le MultimodalSystem
    image_backbone.classifier = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(1024, cfg.get("num_classes", 14)),
    )

    multimodal_model = MultimodalSystem(
        image_backbone=image_backbone,
        text_encoder=tfidf_enc,
        img_out_dim=cfg.get("num_classes", 14),
        txt_out_dim=cfg.get("num_classes", 14),
        num_classes=cfg.get("num_classes", 14),
        fusion_strategy=cfg.get("fusion_strategy", "late"),
        hidden_dim=cfg.get("fusion_hidden_dim", 128),
    )

    # pos_weight approximatif pour OpenI (train split)
    n_pos = labels_array[splits["train"]].sum(axis=0)
    n_neg = len(splits["train"]) - n_pos
    pos_weight = torch.tensor(
        np.where(n_pos > 0, n_neg / np.maximum(n_pos, 1), 1.0), dtype=torch.float32
    )

    exp_name = f"{cfg['mlflow_experiment_prefix']}-multimodal"
    modes_to_run = (
        ["image_only", "text_only", "fusion"] if args.mode == "all" else [args.mode]
    )

    results = {}
    for mode in modes_to_run:
        run_id, auc = run_experiment(
            mode, multimodal_model, tfidf_enc,
            train_loader, val_loader, test_loader,
            cfg, pos_weight, exp_name,
        )
        results[mode] = {"run_id": run_id, "auc_macro": auc}

    if results:
        best_mode = max(results, key=lambda k: results[k]["auc_macro"])
        best_run_id = results[best_mode]["run_id"]
        with mlflow.start_run(run_id=best_run_id):
            mark_best_run(best_run_id, "multimodal")

    print("\nRésumé multimodal :")
    for mode, res in results.items():
        print(f"  {mode:15s} | AUC={res['auc_macro']:.4f}")


def _collate(batch):
    """Collate function pour gérer les textes (str) dans DataLoader."""
    imgs   = torch.stack([b[0] for b in batch])
    texts  = [b[1] for b in batch]
    labels = torch.stack([b[2] for b in batch])
    return imgs, texts, labels


if __name__ == "__main__":
    main()
