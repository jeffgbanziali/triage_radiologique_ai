"""
Script de telechargement des datasets.

Usage :
  python data/download_datasets.py --dataset chestmnist
  python data/download_datasets.py --dataset openi
  python data/download_datasets.py --dataset all
"""
import argparse
import os
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"


def download_chestmnist(resolution: int = 64) -> None:
    """ChestMNIST est telecharge automatiquement par medmnist lors du premier chargement."""
    print(f"\n[ChestMNIST] Telechargement via medmnist (resolution {resolution}px)...")
    try:
        from medmnist import ChestMNIST
        chest_dir = RAW / "chestmnist"
        chest_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            ChestMNIST(split=split, size=resolution, download=True, root=str(chest_dir))
            print(f"  [OK] split '{split}' telecharge.")
        print("[ChestMNIST] Termine.")
    except ImportError:
        print("ERREUR : medmnist non installe. Lancez d'abord : pip install medmnist")


def download_openi() -> None:
    """
    Telecharge le dataset Indiana University Chest X-Rays (OpenI).

    Option A - via Kaggle CLI (recommandee) :
      1. Installez kaggle : pip install kaggle
      2. Configurez votre API key (~/.kaggle/kaggle.json)
      3. Relancez ce script.

    Option B - telechargement manuel :
      1. Allez sur Kaggle : raddar/chest-xrays-indiana-university
      2. Telechargez le zip et extrayez-le dans data/raw/openi/
         Structure attendue :
           data/raw/openi/
             images/images_normalized/  : images PNG
             ecgen-radiology/           : rapports XML
    """
    openi_dir = RAW / "openi"
    openi_dir.mkdir(parents=True, exist_ok=True)

    if (openi_dir / "images" / "images_normalized").exists():
        imgs = list((openi_dir / "images" / "images_normalized").glob("*.png"))
        print(f"[OpenI] Deja present - {len(imgs)} images trouvees.")
        return

    print("\n[OpenI] Tentative de telechargement via Kaggle API...")
    try:
        import kaggle  # type: ignore
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "raddar/chest-xrays-indiana-university",
            path=str(openi_dir),
            unzip=True,
        )
        print("[OpenI] Telechargement Kaggle termine.")
    except ImportError:
        _print_manual_instructions(openi_dir)
    except Exception as e:
        print(f"[OpenI] Kaggle API echoue ({e}).")
        _print_manual_instructions(openi_dir)


def _print_manual_instructions(openi_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("TELECHARGEMENT MANUEL REQUIS pour OpenI :")
    print("  1. Allez sur Kaggle (compte gratuit necessaire) :")
    print("     https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university")
    print("  2. Cliquez 'Download', extrayez le zip dans :")
    print(f"     {openi_dir}")
    print("  Structure attendue apres extraction :")
    print("     data/raw/openi/")
    print("       images/images_normalized/*.png")
    print("       ecgen-radiology/*.xml")
    print("=" * 60)


def verify_datasets() -> None:
    """Verifie que les datasets sont bien presents et affiche un resume."""
    print("\n[Verification des datasets]")

    chest_dir = RAW / "chestmnist"
    if chest_dir.exists():
        npz_files = list(chest_dir.glob("*.npz"))
        status = "OK" if npz_files else "MANQUANT"
        print(f"  ChestMNIST : [{status}] {len(npz_files)} fichier(s) .npz")
    else:
        print("  ChestMNIST : [MANQUANT]")

    openi_img_dir = RAW / "openi" / "images" / "images_normalized"
    if openi_img_dir.exists():
        imgs = list(openi_img_dir.glob("*.png"))
        status = "OK" if imgs else "VIDE"
        print(f"  OpenI images : [{status}] {len(imgs)} images")
    else:
        print("  OpenI images : [MANQUANT]")

    openi_xml_dir = RAW / "openi" / "ecgen-radiology"
    if openi_xml_dir.exists():
        xmls = list(openi_xml_dir.glob("*.xml"))
        status = "OK" if xmls else "VIDE"
        print(f"  OpenI rapports : [{status}] {len(xmls)} fichiers XML")
    else:
        print("  OpenI rapports : [MANQUANT]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telechargement des datasets")
    parser.add_argument("--dataset", default="all", choices=["chestmnist", "openi", "all"])
    parser.add_argument("--resolution", type=int, default=64, choices=[28, 64, 128, 224])
    args = parser.parse_args()

    if args.dataset in ("chestmnist", "all"):
        download_chestmnist(args.resolution)
    if args.dataset in ("openi", "all"):
        download_openi()

    verify_datasets()
