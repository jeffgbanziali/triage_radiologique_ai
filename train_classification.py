"""
Entrainement des 3 architectures de classification multi-label.

Lance en sequence : CNN from scratch, DenseNet121 (transfer learning, 2 phases), DeiT-tiny.
Chaque run est logue dans MLflow. Le meilleur modele par AUC est marque best_model=true.

Usage :
  python train_classification.py
  python train_classification.py --model cnn_scratch
  python train_classification.py --model densenet
  python train_classification.py --model vit
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

_MLFLOW_DB = ROOT / "mlflow.db"
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_MLFLOW_DB}"

import torch
import mlflow

from src.setup_check import merge_configs, set_global_seeds
from src.data.dataset import ChestMNISTDataset
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.splits import make_multilabel_splits, load_splits
from src.models.cnn_from_scratch import build_cnn_from_scratch
from src.models.transfer_learning import build_densenet121
from src.models.vit import build_vit
from src.training.trainer import Trainer, build_optimizer, build_scheduler, build_criterion
from src.training.mlflow_utils import start_run, log_config, log_model, log_figure, mark_best_run
from src.evaluation.metrics import compute_metrics, print_metrics, plot_confusion_matrices


def get_dataloaders(cfg: dict):
    resolution = cfg["resolution"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]
    data_dir = cfg["raw_dir"] + "chestmnist"

    train_ds = ChestMNISTDataset("train", resolution, get_train_transforms(resolution), data_dir)
    val_ds   = ChestMNISTDataset("val",   resolution, get_val_transforms(resolution),   data_dir)
    test_ds  = ChestMNISTDataset("test",  resolution, get_val_transforms(resolution),   data_dir)

    print(f"Dataset - train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    pos_weight = train_ds.get_pos_weight()
    return train_loader, val_loader, test_loader, pos_weight


def train_single_model(model_name: str, cfg_base: dict, pos_weight: torch.Tensor,
                        train_loader, val_loader, test_loader):
    """Entraine un modele et retourne l'AUC macro sur le test set."""
    device = cfg_base["device"]
    exp_name = f"{cfg_base['mlflow_experiment_prefix']}-classification"

    if model_name == "cnn_scratch":
        cfg = merge_configs("base", "cnn_scratch")
        model = build_cnn_from_scratch(cfg)
        run_name = "CNN_from_scratch"
    elif model_name == "densenet":
        cfg = merge_configs("base", "transfer")
        model = build_densenet121(cfg)
        run_name = "DenseNet121_transfer"
    elif model_name == "vit":
        cfg = merge_configs("base", "vit")
        model = build_vit(cfg)
        run_name = "DeiT_tiny_ViT"
    else:
        raise ValueError(f"Modele inconnu : {model_name}")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*55}")
    print(f"Modele : {run_name}")
    print(f"Parametres : {n_params:,} total | {n_trainable:,} entrainables")
    print(f"{'='*55}")

    with start_run(exp_name, run_name) as run:
        log_config(cfg)
        mlflow.log_param("n_params", n_params)
        mlflow.log_param("n_trainable_params", n_trainable)
        mlflow.set_tag("model_type", model_name)

        criterion = build_criterion(cfg, pos_weight.to(device))

        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg, cfg.get("max_epochs", 20))

        trainer = Trainer(model, optimizer, criterion, scheduler, device, cfg)
        history = trainer.fit(
            train_loader, val_loader,
            max_epochs=cfg.get("max_epochs", 20),
            patience=cfg.get("patience", 5),
            save_path=f"data/processed/{model_name}_best.pt",
            mlflow_run=run,
        )

        if model_name == "densenet" and hasattr(model, "unfreeze_last_block"):
            print("\nPhase 2 DenseNet - degel du denseblock4 pour fine-tuning...")
            model.unfreeze_last_block("denseblock4")
            unfreeze_lr = cfg.get("unfreeze_lr", 1e-4)
            optimizer2 = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=unfreeze_lr, weight_decay=cfg.get("weight_decay", 1e-4)
            )
            scheduler2 = build_scheduler(optimizer2, cfg, 10)
            trainer2 = Trainer(model, optimizer2, criterion, scheduler2, device, cfg)
            trainer2.fit(train_loader, val_loader, max_epochs=3, patience=2,
                         save_path=f"data/processed/{model_name}_finetuned.pt",
                         mlflow_run=run)

        val_loss, probs, labels = trainer.val_epoch(test_loader)
        metrics = compute_metrics(labels.numpy(), probs.numpy())
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()
                            if isinstance(v, float)})
        print_metrics(metrics)

        fig = plot_confusion_matrices(labels.numpy(), probs.numpy())
        log_figure(fig, f"confusion_matrix_{model_name}.png")
        import matplotlib.pyplot as plt; plt.close(fig)

        log_model(model, artifact_name=f"model_{model_name}")

        run_id = run.info.run_id
        auc = metrics.get("auc_macro", 0.0)
        print(f"\nRun ID MLflow : {run_id}")
        print(f"AUC macro test : {auc:.4f}")

    return run_id, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=["cnn_scratch", "densenet", "vit", "all"])
    args = parser.parse_args()

    cfg = merge_configs("base")
    set_global_seeds(cfg["seed"])

    train_loader, val_loader, test_loader, pos_weight = get_dataloaders(cfg)

    models_to_train = (
        ["cnn_scratch", "densenet", "vit"] if args.model == "all"
        else [args.model]
    )

    results = {}
    for model_name in models_to_train:
        run_id, auc = train_single_model(
            model_name, cfg, pos_weight, train_loader, val_loader, test_loader
        )
        results[model_name] = {"run_id": run_id, "auc_macro": auc}

    if results:
        best_name = max(results, key=lambda k: results[k]["auc_macro"])
        best_run_id = results[best_name]["run_id"]
        with mlflow.start_run(run_id=best_run_id):
            mark_best_run(best_run_id, "classification", artifact_key=f"model_{best_name}")
        print(f"\nMeilleur modele : {best_name} (AUC={results[best_name]['auc_macro']:.4f})")

    print("\nResume des runs :")
    for name, res in results.items():
        print(f"  {name:15s} | AUC={res['auc_macro']:.4f} | run_id={res['run_id']}")


if __name__ == "__main__":
    main()
