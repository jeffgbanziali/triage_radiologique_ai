"""Helpers MLflow pour le tracking des experiences (classification, anomaly, multimodal)."""
import os
from pathlib import Path
from typing import Any, Dict, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MLFLOW_DB_PATH = _PROJECT_ROOT / "mlflow.db"
MLFLOW_DB_URI = f"sqlite:///{MLFLOW_DB_PATH}"

os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_DB_URI

import mlflow
import mlflow.pytorch

mlflow.set_tracking_uri(MLFLOW_DB_URI)


def get_or_create_experiment(experiment_name: str) -> str:
    """Retourne l'ID de l'experience MLflow, la cree si necessaire."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return mlflow.create_experiment(experiment_name)
    return exp.experiment_id


def start_run(experiment_name: str, run_name: str, tags: Optional[Dict] = None):
    """Demarre un run MLflow dans l'experience donnee."""
    exp_id = get_or_create_experiment(experiment_name)
    return mlflow.start_run(experiment_id=exp_id, run_name=run_name, tags=tags or {})


def log_config(cfg: dict) -> None:
    """Logue tous les hyperparametres de la config dans MLflow."""
    flat = _flatten_dict(cfg)
    params = {
        k: str(v)[:250]
        for k, v in flat.items()
        if v is not None and not isinstance(v, (list, dict))
    }
    mlflow.log_params(params)


def log_metrics_epoch(metrics: Dict[str, float], step: int) -> None:
    """Logue les metriques d'une epoch dans MLflow."""
    mlflow.log_metrics(
        {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        step=step,
    )


def log_model(model, artifact_name: str = "model") -> None:
    """Logue le modele PyTorch comme artefact MLflow."""
    mlflow.pytorch.log_model(
        model,
        artifact_path=artifact_name,
        serialization_format="pickle",
    )


def log_figure(fig, filename: str) -> None:
    """Logue une figure matplotlib comme artefact MLflow (figures/)."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, dpi=100, bbox_inches="tight")
        mlflow.log_artifact(tmp.name, artifact_path="figures")
    Path(tmp.name).unlink(missing_ok=True)


def log_array_as_artifact(arr, filename: str) -> None:
    """Sauvegarde un numpy array comme artefact MLflow (.npy)."""
    import numpy as np
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        np.save(tmp.name, arr)
        mlflow.log_artifact(tmp.name, artifact_path="arrays")
    Path(tmp.name).unlink(missing_ok=True)


def mark_best_run(run_id: str, component: str, artifact_key: Optional[str] = None) -> None:
    """Marque un run comme best_model=true et met a jour configs/base.yaml."""
    mlflow.set_tag("best_model", "true")
    mlflow.set_tag("component", component)

    try:
        import yaml
        cfg_path = Path("configs/base.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("best_run_ids", {})[component] = run_id
        if artifact_key:
            cfg.setdefault("best_artifact_keys", {})[component] = artifact_key
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        print(f"Run {run_id} marque comme meilleur modele pour '{component}'.")
    except Exception as e:
        print(f"Avertissement : impossible de mettre a jour base.yaml -- {e}")


def _flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    """Aplatit un dictionnaire imbrique pour MLflow log_params."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep))
        else:
            items[new_key] = v
    return items
