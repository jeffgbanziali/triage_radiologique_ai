---
title: Triage Radiologique TRI-AI
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.0
app_file: demo/app.py
pinned: false
---

# Triage Radiologique - Projet Deep Learning EFREI 2026

Systeme d'aide au tri de radiographies thoraciques combinant classification multi-label,
detection d'anomalies et fusion multimodale image-texte.

---

## Auteurs

Projet realise dans le cadre du cours Machine Learning et Deep Learning — EFREI Paris, 2026.

| Nom | GitHub |
|-----|--------|
| Jeff GBANZIALI | [@Jeffflaj](https://github.com/Jeffflaj) |
| Franck MBOUTOU | — |
| Emilie DOUAN HOAI-HUONG | — |

---

## Contexte

L'objectif est de construire un pipeline complet : preparation des donnees, entrainement
de plusieurs architectures, suivi des experiences avec MLflow, et demonstration interactive
via Streamlit.

---

## Datasets

### ChestMNIST (NIH ChestX-ray14 via MedMNIST)

| Split | Images |
|-------|--------|
| Train | 78 468 |
| Val   | 11 219 |
| Test  | 22 433 |

- Resolution : 64 x 64 pixels, 3 canaux (RGB)
- 14 pathologies thoraciques en classification multi-label binaire
- Classes : Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule,
  Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis,
  Pleural_Thickening, Hernia
- Source originale : Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks", CVPR 2017
- Distribution via MedMNIST v2 : Yang et al., Scientific Data 2023 — https://medmnist.com/

### OpenI - Indiana University Chest X-Rays

- 3 818 paires image / compte-rendu radiologique (vues frontales)
- Utilise pour la composante multimodale uniquement
- Source : Indiana University Chest X-Ray Collection, Demner-Fushman et al., J Am Med Inform Assoc 2016
- Distribution Kaggle : https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university

---

## Resultats

### Classification (test set ChestMNIST, 22 433 images)

| Modele | Parametres | AUC macro | F1 macro | MCC macro | Bal. Acc. |
|--------|-----------|-----------|---------|-----------|-----------|
| CNN from scratch | ~1.2M | baseline | baseline | baseline | baseline |
| DenseNet121 (transfer learning) | 7.9M | 0.7122 | 0.1556 | N/A | N/A |
| DeiT-tiny (Vision Transformer) | 5.7M | **0.7374** | **0.1659** | **0.1435** | **0.6478** |

Le DeiT-tiny est le meilleur modele et est utilise par defaut dans la demonstration.

### Detection d'anomalies

Autoencoder convolutionnel entraine uniquement sur les images normales (42 405 images).

- Score d'anomalie : erreur de reconstruction MSE
- Seuil (percentile 95 du val set normal) : 0.00511
- Taux d'anomalie sur le test set : 4.6 %

### Fusion multimodale (OpenI, 573 images test)

| Mode | AUC macro |
|------|----------|
| image_only | 0.6483 |
| fusion (late fusion) | 0.9662 |
| text_only (TF-IDF) | **0.9727** |

Le mode texte seul (TF-IDF) surpasse la fusion car les comptes-rendus OpenI
contiennent les diagnostics de facon explicite dans le texte.

---

## Installation

### Prerequis

- Python 3.10 ou 3.11
- pip
- Git + Git LFS
- Pas de GPU requis (CPU suffisant)

### 1. Cloner le depot

```bash
git clone https://github.com/Jeffflaj/deep_learning_project.git
cd triage-radiologique
```

### 2. Creer un environnement virtuel

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Installer PyTorch CPU

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Installer les dependances

```bash
pip install -r requirements.txt
```

### 5. Verifier l'environnement

```bash
python src/setup_check.py
```

---

## Telecharger les donnees

### ChestMNIST (automatique)

```bash
python data/download_datasets.py
```

ChestMNIST est telecharge automatiquement via medmnist dans `data/raw/`.

### OpenI (Kaggle)

1. Creer un compte sur [kaggle.com](https://www.kaggle.com)
2. Aller dans Account -> Create New Token -> telecharger `kaggle.json`
3. Placer le fichier :
   - Windows : `C:\Users\<user>\.kaggle\kaggle.json`
   - Linux/macOS : `~/.kaggle/kaggle.json`
4. Installer le client Kaggle :

```bash
pip install kaggle
```

5. Telecharger le dataset :

```bash
kaggle datasets download raddar/chest-xrays-indiana-university -p data/ --unzip
```

---

## Entrainer les modeles

```bash
# Classification (CNN, DenseNet121, DeiT-tiny)
python train_classification.py

# Un seul modele
python train_classification.py --model vit

# Detection d'anomalies (autoencoder convolutionnel)
python train_anomaly.py

# Fusion multimodale sur OpenI
python train_multimodal.py
```

Les checkpoints sont sauvegardes dans `data/processed/` et les experiences
sont loguees dans `mlflow.db`.

---

## Lancer la demonstration Streamlit

### Option A — Telecharger les modeles pre-entraines (recommande)

Les modeles sont heberges sur Hugging Face Hub. Un seul script suffit :

```bash
python download_models.py
streamlit run demo/app.py
```

### Option B — Reentrainer depuis zero

```bash
python train_classification.py --model vit
python train_anomaly.py
streamlit run demo/app.py
```

Interface accessible sur http://localhost:8501. Trois pages :

- **Analyse** : chargement d'une image, inference (classification + anomalie),
  carte d'attention Attention Rollout, priorite de triage (URGENT / SURVEILLANCE / NORMAL)
- **Dashboard** : comparaison des modeles depuis MLflow, metriques par classe
- **A propos** : description du projet, architectures, resultats

---

## Consulter MLflow

MLflow permet de visualiser toutes les experiences, comparer les modeles et acceder
aux artefacts (courbes de loss, matrices de confusion, checkpoints).

### Lancer l'interface MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Interface disponible sur http://localhost:5000.

### Via l'API MLflow (Python)

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()

# Lister toutes les experiences
experiments = client.search_experiments()
for exp in experiments:
    print(exp.name)

# Recuperer le meilleur run d'une experience
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["metrics.test_auc_macro DESC"],
    max_results=1
)
best_run = runs[0]
print(f"AUC : {best_run.data.metrics['test_auc_macro']:.4f}")
print(f"Run ID : {best_run.info.run_id}")

# Charger un modele sauvegarde
import torch
model_path = f"data/processed/vit_best.pt"
state_dict = torch.load(model_path, map_location="cpu")
```

---

## Tests

```bash
pytest tests/ -v
```

---

## Architectures

### CNN from scratch

Baseline sans transfer learning. 4 blocs Conv-BN-ReLU-MaxPool, tete MLP 512 -> 14 classes,
environ 1.2M de parametres.

### DenseNet121 - Transfer learning

Backbone pre-entraine sur ImageNet. Transfer learning en deux phases :
- Phase 1 : backbone gele, seule la tete s'entraine
- Phase 2 : fine-tuning du dernier bloc dense (denseblock4) avec LR reduit x10

### DeiT-tiny - Vision Transformer

DeiT-tiny de timm, pre-entraine par distillation depuis un CNN enseignant.
Image 64x64 decoupee en 16 patchs de 16x16 pixels (4x4 grille spatiale).
Entrainement avec AdamW, lr=5e-5, CosineAnnealingLR, 5 epochs.

La carte d'attention (Attention Rollout) est generee en chaine-multipliant les matrices
d'attention des 12 couches avec un terme residuel identite, donnant une carte 4x4
representant les zones de l'image utilisees pour la prediction.

### Autoencoder convolutionnel

- Encodeur : 4 blocs Conv(stride=2), 32->64->128->256 canaux, espace latent de 128 dimensions
- Decodeur : miroir (ConvTranspose2d)
- Entraine uniquement sur les images normales

---

## Structure du projet

```
triage-radiologique/
|-- configs/               # Hyperparametres par modele (YAML)
|   |-- base.yaml
|   |-- cnn_scratch.yaml
|   |-- transfer.yaml      # DenseNet121
|   |-- vit.yaml           # DeiT-tiny
|   |-- anomaly.yaml
|   `-- multimodal.yaml
|
|-- data/
|   |-- download_datasets.py
|   |-- raw/               # Donnees brutes (non versionne)
|   |-- processed/         # Checkpoints des modeles (.pt)
|   `-- splits/            # Indices train/val/test (JSON)
|
|-- demo/
|   |-- app.py             # Page principale Streamlit
|   `-- pages/
|       |-- 1_Dashboard.py
|       `-- 2_Apropos.py
|
|-- src/
|   |-- anomaly/
|   |-- data/
|   |-- evaluation/
|   |-- models/
|   |-- multimodal/
|   `-- training/
|
|-- tests/
|   `-- test_models.py
|
|-- train_classification.py
|-- train_anomaly.py
|-- train_multimodal.py
`-- requirements.txt
```

---

## Stack technique

| Categorie | Outils |
|-----------|--------|
| Deep Learning | PyTorch, timm, torchvision |
| Tracking | MLflow, SQLite |
| Interface | Streamlit, Plotly |
| Donnees | medmnist, scikit-learn, pandas, numpy |
| Tests | pytest |

---

## Avertissement

Ce projet est realise a des fins academiques uniquement. Il ne constitue pas un
dispositif medical certifie et ne doit pas etre utilise pour des decisions cliniques reelles.
