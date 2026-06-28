"""
Telecharge les modeles pre-entraines depuis Hugging Face Hub.
A lancer une seule fois avant de demarrer la demonstration.

Usage : python download_models.py
"""
from pathlib import Path
import urllib.request

HF_BASE = "https://huggingface.co/Jeffflaj/triage-radiologique-models/resolve/main"

MODELS = {
    "vit_best.pt":           f"{HF_BASE}/vit_best.pt",
    "densenet_best.pt":      f"{HF_BASE}/densenet_best.pt",
    "anomaly_ae_conv_best.pt": f"{HF_BASE}/anomaly_ae_conv_best.pt",
}

def download_models():
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in MODELS.items():
        dest = out_dir / filename
        if dest.exists():
            print(f"[OK] {filename} deja present.")
            continue
        print(f"Telechargement de {filename}...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"[OK] {filename} telecharge.")
        except Exception as e:
            print(f"[ERREUR] {filename} : {e}")
            print(f"  -> Telechargez manuellement depuis : {url}")

if __name__ == "__main__":
    download_models()
    print("\nTous les modeles sont prets. Lancez : streamlit run demo/app.py")
