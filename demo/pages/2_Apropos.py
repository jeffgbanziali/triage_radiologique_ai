import streamlit as st

st.set_page_config(page_title="A propos - TRI-AI", layout="wide")

st.markdown("""
<style>
  .stApp { background-color: #0d1117; color: #e6edf3; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:12px; padding:1rem 1.4rem; margin-bottom:1rem; }
  h1,h2,h3 { color:#58a6ff !important; }
  code { background:#21262d; padding:.1rem .4rem; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

st.markdown("# A propos du projet TRI-AI")
st.markdown("**Projet EFREI - Machine Learning & Deep Learning**")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("## Objectif")
    st.markdown("""
Systeme d'aide au tri radiologique combinant :
- Classification multi-label de 14 pathologies thoraciques
- Detection de cas atypiques hors distribution
- Fusion multimodale image + compte-rendu textuel
- Interface de demonstration interactive
    """)

    st.markdown("## Datasets")
    st.markdown("""
**ChestMNIST** (NIH ChestX-ray14 via medmnist)
- 112 120 radiographies thoraciques a 64x64 px
- 14 pathologies binaires (multi-label)
- Split : 78 468 train / 11 219 val / 22 433 test

**OpenI** (Indiana University CXR)
- 3 818 paires image-compte-rendu
- Utilise pour la composante multimodale
    """)

    st.markdown("## Architectures")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
**CNN from scratch** - Baseline
- 4 blocs Conv-BN-ReLU-MaxPool, tete MLP 512->14
- ~1.2M parametres

**DenseNet121** - Transfer learning
- Backbone pre-entraine ImageNet (7.9M params)
- Phase 1 : backbone gele, tete seule
- Phase 2 : fine-tuning denseblock4
- AUC macro test = **0.7122**

**DeiT-tiny** - Vision Transformer (meilleur modele)
- 5.7M parametres, 12 couches d'attention
- 16 patchs de 16x16 sur image 64x64
- LR = 5e-5, AdamW, CosineAnnealing
- AUC macro test = **0.7374**
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("## Detection d'anomalies")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
**Autoencoder convolutionnel (AE)**
- Encodeur : 4 blocs Conv(stride=2), 32->64->128->256 canaux
- Espace latent : 128 dimensions
- Decodeur : miroir (ConvTranspose2d)
- Entraine uniquement sur les images normales (42 405 images)
- Score d'anomalie = MSE de reconstruction
- Seuil = percentile 95 du val set normal : **0.00511**
- Taux d'anomalie test = **4.6%**
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("## Fusion multimodale")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
**3 modes compares sur OpenI :**

| Mode | AUC test |
|------|---------|
| image_only | 0.6483 |
| fusion (late) | 0.9662 |
| text_only (meilleur) | **0.9727** |

Le TF-IDF seul surpasse la fusion car les comptes-rendus
OpenI contiennent les diagnostics explicitement.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("## Attention Rollout (DeiT)")
    st.markdown("""
1. Extraction des poids d'attention des 12 couches
2. Moyenne sur les tetes d'attention
3. Ajout des connexions residuelles (A + I)
4. Produit matriciel en chaine (rollout global)
5. Carte 4x4 upsampled a la resolution de l'image
    """)

st.divider()
st.markdown("## Stack technique")
c1, c2, c3, c4 = st.columns(4)
c1.markdown("**Deep Learning**\n\nPyTorch · timm · torchvision")
c2.markdown("**Tracking**\n\nMLflow · SQLite")
c3.markdown("**Interface**\n\nStreamlit · Plotly")
c4.markdown("**Data**\n\nmedmnist · scikit-learn · numpy")

st.divider()
st.markdown("""
> **Avertissement** : Ce systeme est developpe a des fins academiques uniquement.
> Il ne constitue pas un dispositif medical certifie et ne doit pas etre utilise pour
> des decisions cliniques reelles.
""")
st.caption("EFREI - Projet Deep Learning · 2026")
