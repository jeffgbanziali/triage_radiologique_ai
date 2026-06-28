"""
Detection automatique du materiel disponible.
Met a jour configs/base.yaml avec les parametres adaptes au hardware.
"""
import os
import sys
import random
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def detect_hardware() -> dict:
    """Detecte GPU/CPU et retourne les parametres d'entrainement adaptes."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU detecte : {gpu_name} ({vram_gb:.1f} GB VRAM)")
            if vram_gb >= 8:
                resolution, batch_size = 224, 32
            elif vram_gb >= 4:
                resolution, batch_size = 128, 16
            else:
                resolution, batch_size = 64, 16
            device = "cuda"
        else:
            print("Aucun GPU CUDA detecte - mode CPU.")
            device = "cpu"
            resolution, batch_size = 64, 16
    except ImportError:
        print("PyTorch non installe - reglages CPU par defaut.")
        device, resolution, batch_size = "cpu", 64, 16
        cuda_available = False

    # num_workers=0 obligatoire sur Windows (pas de fork)
    num_workers = 0 if sys.platform == "win32" else 2

    return {
        "device": device,
        "resolution": resolution,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "cuda_available": cuda_available,
    }


def set_global_seeds(seed: int = 42) -> None:
    """Fixe les seeds pour la reproductibilite des experiences."""
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def update_base_config(hw: dict) -> None:
    """Ecrit les parametres hardware dans configs/base.yaml."""
    config_path = ROOT / "configs" / "base.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["device"] = hw["device"]
    config["resolution"] = hw["resolution"]
    config["batch_size"] = hw["batch_size"]
    config["num_workers"] = hw["num_workers"]

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"configs/base.yaml mis a jour : device={hw['device']}, "
          f"resolution={hw['resolution']}px, batch_size={hw['batch_size']}")


def load_config(name: str = "base") -> dict:
    """Charge un fichier YAML depuis configs/."""
    path = ROOT / "configs" / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base_name: str = "base", override_name: str = None) -> dict:
    """Fusionne base.yaml avec un config specifique (les cles override ecrasent base)."""
    cfg = load_config(base_name)
    if override_name:
        override = load_config(override_name)
        cfg.update(override)
    return cfg


if __name__ == "__main__":
    print("=" * 55)
    print("  Triage Radiologique - Verification de l'environnement")
    print("=" * 55)

    hw = detect_hardware()
    set_global_seeds(42)
    update_base_config(hw)

    print("\nResume :")
    for k, v in hw.items():
        print(f"  {k:20s} : {v}")
    print("\nSeeds fixes (Python, NumPy, PyTorch) - seed=42")
    print("Setup termine.")
