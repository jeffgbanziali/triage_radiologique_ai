"""Boucle d'entrainement avec early stopping, scheduler et MLflow logging."""
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader


class EarlyStopping:
    """Stop si val_loss ne s'ameliore plus pendant `patience` epochs consecutives."""
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            import copy
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def build_optimizer(model: nn.Module, cfg: dict):
    opt_name = cfg.get("optimizer", "adamw").lower()
    lr       = float(cfg.get("lr", 1e-3))
    wd       = float(cfg.get("weight_decay", 1e-4))

    params = filter(lambda p: p.requires_grad, model.parameters())
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Optimiseur inconnu : {opt_name}")


def build_scheduler(optimizer, cfg: dict, num_epochs: int):
    """Scheduler de LR : CosineAnnealingLR ou StepLR selon la config."""
    sched = cfg.get("scheduler", "cosine").lower()
    if sched == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif sched == "step":
        return StepLR(optimizer, step_size=cfg.get("step_size", 7), gamma=0.1)
    return None


def build_criterion(cfg: dict, pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    loss_name = cfg.get("loss", "bce_weighted").lower()
    if "bce" in loss_name:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    raise ValueError(f"Loss inconnue : {loss_name}")


class Trainer:
    """Boucle d'entrainement pour la classification multi-label."""

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        criterion: nn.Module,
        scheduler=None,
        device: str = "cpu",
        cfg: dict = None,
    ):
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device    = device
        self.cfg       = cfg or {}

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(imgs)
            loss   = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader) -> Tuple[float, torch.Tensor, torch.Tensor]:
        self.model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs)
            loss   = self.criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())
        return (
            total_loss / len(loader.dataset),
            torch.cat(all_probs),
            torch.cat(all_labels),
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int,
        patience: int = 5,
        save_path: Optional[str] = None,
        mlflow_run=None,
    ) -> Dict:
        from src.evaluation.metrics import compute_metrics
        from src.training.mlflow_utils import log_metrics_epoch

        early_stop = EarlyStopping(patience=patience)
        history    = {"train_loss": [], "val_loss": [], "val_f1_macro": []}

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, probs, labels = self.val_epoch(val_loader)

            metrics = compute_metrics(
                labels.numpy(), probs.numpy(), threshold=0.5
            )

            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"F1={metrics['f1_macro']:.4f} | MCC={metrics['mcc_macro']:.4f} | "
                f"{elapsed:.1f}s"
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1_macro"].append(metrics["f1_macro"])

            if mlflow_run:
                log_metrics_epoch(
                    {"train_loss": train_loss, "val_loss": val_loss, **metrics},
                    step=epoch,
                )

            if early_stop.step(val_loss, self.model):
                print(f"Early stopping a l'epoch {epoch}.")
                break

        early_stop.restore_best(self.model)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            print(f"Meilleur modele sauvegarde -> {save_path}")

        return history


class AnomalyTrainer:
    """Boucle d'entrainement pour l'AE/VAE."""

    def __init__(self, model: nn.Module, optimizer, device: str = "cpu", is_vae: bool = False):
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.device    = device
        self.is_vae    = is_vae

    def train_epoch(self, loader: DataLoader) -> Dict:
        self.model.train()
        total = {"loss": 0.0, "rec": 0.0, "kl": 0.0}
        n = 0
        for batch in loader:
            imgs = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
            self.optimizer.zero_grad()
            if self.is_vae:
                x_hat, mu, logvar = self.model(imgs)
                loss, parts = self.model.loss(imgs, x_hat, mu, logvar)
                total["rec"] += parts["loss_rec"] * imgs.size(0)
                total["kl"]  += parts["loss_kl"]  * imgs.size(0)
            else:
                x_hat, _ = self.model(imgs)
                import torch.nn.functional as F
                loss = F.mse_loss(x_hat, imgs)
                total["rec"] += loss.item() * imgs.size(0)
            loss.backward()
            self.optimizer.step()
            total["loss"] += loss.item() * imgs.size(0)
            n += imgs.size(0)
        return {k: v / n for k, v in total.items()}

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for batch in loader:
            imgs = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
            if self.is_vae:
                x_hat, mu, logvar = self.model(imgs)
                loss, _ = self.model.loss(imgs, x_hat, mu, logvar)
            else:
                x_hat, _ = self.model(imgs)
                import torch.nn.functional as F
                loss = F.mse_loss(x_hat, imgs)
            total += loss.item() * imgs.size(0)
            n += imgs.size(0)
        return total / n

    def fit(self, train_loader, val_loader, max_epochs=30, patience=7, save_path=None):
        from src.training.mlflow_utils import log_metrics_epoch
        early_stop = EarlyStopping(patience=patience)
        for epoch in range(1, max_epochs + 1):
            train_m = self.train_epoch(train_loader)
            val_loss = self.val_epoch(val_loader)
            print(f"Epoch {epoch:3d}/{max_epochs} | train_loss={train_m['loss']:.5f} | val_loss={val_loss:.5f}")
            if early_stop.step(val_loss, self.model):
                print(f"Early stopping a l'epoch {epoch}.")
                break
        early_stop.restore_best(self.model)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
