import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import streamlit as st
import torch
import yaml
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{ROOT / 'mlflow.db'}"

st.set_page_config(
    page_title="TRI-AI - Triage Radiologique",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp { background-color: #0d1117; color: #e6edf3; }
  .card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
  }
  .badge-urgent {
    display:inline-block; background:#da3633; color:#fff;
    padding:.4rem 1.2rem; border-radius:24px; font-size:1.1rem; font-weight:700;
  }
  .badge-surveillance {
    display:inline-block; background:#d29922; color:#fff;
    padding:.4rem 1.2rem; border-radius:24px; font-size:1.1rem; font-weight:700;
  }
  .badge-normal {
    display:inline-block; background:#238636; color:#fff;
    padding:.4rem 1.2rem; border-radius:24px; font-size:1.1rem; font-weight:700;
  }
  h1 { color:#58a6ff !important; }
  section[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #30363d; }
  [data-testid="stMetricLabel"] { color:#8b949e !important; font-size:.8rem !important; }
  [data-testid="stMetricValue"] { color:#e6edf3 !important; }
  hr { border-color:#30363d; }
  .hist-item {
    background:#21262d; border-radius:8px; padding:.4rem .8rem;
    margin-bottom:.4rem; font-size:.8rem; color:#8b949e;
    border-left:3px solid #58a6ff;
  }
  .stDownloadButton > button {
    background:#238636; color:#fff; border:none;
    border-radius:8px; font-weight:600; width:100%;
  }
  .stDownloadButton > button:hover { background:#2ea043; }
</style>
""", unsafe_allow_html=True)

LABEL_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
LABEL_FR = {
    "Atelectasis": "Atelectasie", "Cardiomegaly": "Cardiomegalie",
    "Effusion": "Epanchement pleural", "Infiltration": "Infiltration",
    "Mass": "Masse", "Nodule": "Nodule", "Pneumonia": "Pneumonie",
    "Pneumothorax": "Pneumothorax", "Consolidation": "Consolidation",
    "Edema": "Oedeme pulmonaire", "Emphysema": "Emphyseme",
    "Fibrosis": "Fibrose", "Pleural_Thickening": "Epaississement pleural",
    "Hernia": "Hernie diaphragmatique",
}
SEVERITY = {
    "Pneumothorax": 1.00, "Pneumonia": 0.88, "Edema": 0.85,
    "Mass": 0.82, "Consolidation": 0.75, "Cardiomegaly": 0.70,
    "Effusion": 0.68, "Nodule": 0.65, "Emphysema": 0.60,
    "Atelectasis": 0.58, "Infiltration": 0.55, "Fibrosis": 0.50,
    "Pleural_Thickening": 0.45, "Hernia": 0.40,
}


@st.cache_resource
def load_cfg():
    with open(ROOT / "configs" / "base.yaml") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_clf_model():
    cfg = load_cfg()
    run_id = cfg.get("best_run_ids", {}).get("classification")
    if not run_id:
        return None, None, "Aucun run de classification trouve."
    artifact_key = cfg.get("best_artifact_keys", {}).get("classification", "model_vit")
    try:
        import mlflow.pytorch
        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
        model = mlflow.pytorch.load_model(f"runs:/{run_id}/{artifact_key}")
        model.eval()
        return model, artifact_key, None
    except Exception as e:
        return None, None, str(e)


@st.cache_resource
def load_anom_model():
    cfg = load_cfg()
    run_id = cfg.get("best_run_ids", {}).get("anomaly")
    if not run_id:
        return None, 0.005, "Aucun run anomalie trouve."
    try:
        import mlflow, mlflow.pytorch
        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
        artifact_key = cfg.get("best_artifact_keys", {}).get("anomaly", "model_ae_conv")
        model = mlflow.pytorch.load_model(f"runs:/{run_id}/{artifact_key}")
        model.eval()
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        threshold = float(run.data.metrics.get("anomaly_threshold", 0.005))
        return model, threshold, None
    except Exception as e:
        return None, 0.005, str(e)


def preprocess(img: Image.Image, resolution: int, normalize: bool = True):
    img = img.convert("RGB")
    ops = [transforms.Resize((resolution, resolution)), transforms.ToTensor()]
    if normalize:
        ops.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(ops)(img).unsqueeze(0)


@torch.no_grad()
def predict(model, tensor):
    return torch.sigmoid(model(tensor)).squeeze().numpy()


@torch.no_grad()
def anomaly_score(model, tensor):
    try:
        x_hat, _ = model(tensor)
    except TypeError:
        x_hat, _, _ = model(tensor)
    score = ((tensor - x_hat) ** 2).mean().item()
    recon = x_hat.squeeze().permute(1, 2, 0).numpy().clip(0, 1)
    return score, recon


def attention_rollout(model, tensor):
    attentions = []
    hooks = []

    def _hook(module, inp, out):
        attentions.append(out.detach().cpu())

    backbone = getattr(model, "backbone", None)
    if backbone is None or not hasattr(backbone, "blocks"):
        return None

    for block in backbone.blocks:
        if hasattr(block, "attn") and hasattr(block.attn, "attn_drop"):
            hooks.append(block.attn.attn_drop.register_forward_hook(_hook))

    with torch.no_grad():
        model(tensor)

    for h in hooks:
        h.remove()

    if not attentions:
        return None

    # Rollout : produit des matrices d'attention avec connexions residuelles
    result = torch.eye(attentions[0].shape[-1])
    for attn in attentions:
        attn_mean = attn[0].mean(dim=0)
        attn_res = attn_mean + torch.eye(attn_mean.shape[-1])
        attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True)
        result = attn_res @ result

    mask = result[0, 1:].numpy()
    side = int(np.sqrt(len(mask)))
    mask = mask.reshape(side, side)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


def overlay_heatmap(img: Image.Image, mask: np.ndarray, alpha: float = 0.5):
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    heat = cm.get_cmap("jet")(np.array(mask_img) / 255.0)[:, :, :3]
    orig = np.array(img.convert("RGB")) / 255.0
    blended = (1 - alpha) * orig + alpha * heat
    return Image.fromarray((blended * 255).clip(0, 255).astype(np.uint8))


def compute_triage(probs, threshold, is_anomaly):
    weighted = [
        (LABEL_NAMES[i], float(probs[i]), float(probs[i]) * SEVERITY[LABEL_NAMES[i]])
        for i in range(len(LABEL_NAMES)) if probs[i] >= threshold
    ]
    max_sev = max((w for _, _, w in weighted), default=0.0)

    if is_anomaly or max_sev >= 0.45:
        level, css = "URGENT", "badge-urgent"
    elif max_sev >= 0.20 or weighted:
        level, css = "SURVEILLANCE", "badge-surveillance"
    else:
        level, css = "NORMAL", "badge-normal"

    return level, css, max_sev, sorted(weighted, key=lambda x: -x[2])


def chart_classification(probs, threshold):
    import plotly.graph_objects as go
    labels = [LABEL_FR[l] for l in LABEL_NAMES]
    colors = ["#da3633" if p >= threshold else "#388bfd" for p in probs]
    idx = np.argsort(probs)
    fig = go.Figure(go.Bar(
        x=[probs[i] for i in idx],
        y=[labels[i] for i in idx],
        orientation="h",
        marker_color=[colors[i] for i in idx],
        text=[f"{probs[i]:.1%}" for i in idx],
        textposition="outside",
        textfont=dict(color="white", size=11),
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#d29922",
                  annotation_text=f"Seuil {threshold:.0%}",
                  annotation_font_color="#d29922")
    fig.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
        font_color="#e6edf3", margin=dict(l=10, r=70, t=10, b=10),
        xaxis=dict(range=[0, 1.15], gridcolor="#21262d", tickformat=".0%"),
        yaxis=dict(gridcolor="#21262d"),
        height=420, showlegend=False,
    )
    return fig


def chart_anomaly_gauge(score, threshold):
    import plotly.graph_objects as go
    max_val = max(threshold * 3, score * 1.5)
    color = "#da3633" if score > threshold else "#238636"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"color": color, "size": 28}, "valueformat": ".5f"},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#8b949e",
                     "tickformat": ".4f", "tickfont": {"color": "#8b949e"}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#161b22", "borderwidth": 0,
            "steps": [
                {"range": [0, threshold], "color": "#1a2a1a"},
                {"range": [threshold, max_val], "color": "#2a1a1a"},
            ],
            "threshold": {
                "line": {"color": "#d29922", "width": 3},
                "thickness": 0.85, "value": threshold,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#161b22", font_color="#e6edf3",
        height=220, margin=dict(l=20, r=20, t=20, b=10),
    )
    return fig


def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []


def add_history(filename, level, top_label):
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "file": filename,
        "level": level,
        "top": top_label,
    })
    st.session_state.history = st.session_state.history[:10]


def main():
    init_state()
    cfg = load_cfg()
    resolution = cfg.get("resolution", 64)

    with st.sidebar:
        st.markdown("## Parametres")
        threshold = st.slider("Seuil de detection", 0.10, 0.90, 0.40, 0.05)
        show_gradcam = st.checkbox("Carte d'attention (DeiT)", value=True)
        show_recon   = st.checkbox("Reconstruction AE", value=False)
        st.divider()

        st.markdown("## Historique session")
        if st.session_state.history:
            for h in st.session_state.history:
                color = {"URGENT": "#da3633", "SURVEILLANCE": "#d29922",
                         "NORMAL": "#238636"}.get(h["level"], "#58a6ff")
                st.markdown(
                    f'<div class="hist-item">'
                    f'<span style="color:{color};font-weight:700">{h["level"]}</span> '
                    f'· {h["time"]}<br><small>{h["file"][:28]}</small></div>',
                    unsafe_allow_html=True,
                )
            if st.button("Effacer l'historique", use_container_width=True):
                st.session_state.history = []
                st.rerun()
        else:
            st.caption("Aucune analyse pour l'instant.")

        st.divider()
        run_ids = cfg.get("best_run_ids", {})
        artifact_keys = cfg.get("best_artifact_keys", {})
        st.markdown("**Modeles actifs**")
        st.caption(f"Classification : `{artifact_keys.get('classification', '?')}`")
        st.caption(f"Anomalie       : `{artifact_keys.get('anomaly', '?')}`")
        clf_runid = run_ids.get("classification", "")
        anom_runid = run_ids.get("anomaly", "")
        st.caption(f"Run CLF : `{clf_runid[:8] if clf_runid else 'N/A'}`")
        st.caption(f"Run AE  : `{anom_runid[:8] if anom_runid else 'N/A'}`")

    st.markdown("# TRI-AI - Triage Radiologique")
    st.markdown(
        "Systeme d'aide a la decision base sur CNN/ViT + detection d'anomalies.  "
        "**Usage academique uniquement - ne constitue pas un avis medical.**"
    )
    st.divider()

    clf_model, artifact_key, clf_err = load_clf_model()
    anom_model, anom_threshold, anom_err = load_anom_model()

    col_up, col_txt = st.columns([1, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Charger une radiographie thoracique",
            type=["png", "jpg", "jpeg"],
        )
    with col_txt:
        report_text = st.text_area(
            "Compte-rendu radiologique (optionnel)",
            height=100,
            placeholder="Ex: Bilateral lower lobe consolidation consistent with pneumonia...",
        )

    if uploaded is None:
        _welcome()
        return

    img = Image.open(uploaded).convert("RGB")
    tensor_clf  = preprocess(img, resolution, normalize=True)
    tensor_anom = preprocess(img, resolution, normalize=False)

    probs = predict(clf_model, tensor_clf) if clf_model else np.zeros(14)
    score_anom, recon = anomaly_score(anom_model, tensor_anom) if anom_model else (0.0, None)
    is_anomaly = score_anom > anom_threshold

    level, badge_css, sev_score, top_findings = compute_triage(probs, threshold, is_anomaly)

    heatmap = None
    if show_gradcam and clf_model is not None and artifact_key and "vit" in artifact_key:
        heatmap = attention_rollout(clf_model, tensor_clf)

    top_label = LABEL_FR[top_findings[0][0]] if top_findings else "RAS"
    add_history(uploaded.name, level, top_label)

    st.markdown("---")
    st.markdown(
        f'<div style="text-align:center;margin-bottom:1rem">'
        f'<span class="{badge_css}">{level}</span>'
        f'<span style="color:#8b949e;margin-left:1rem;font-size:.9rem">'
        f'Score de severite : {sev_score:.2f}</span></div>',
        unsafe_allow_html=True,
    )

    col_img, col_res = st.columns([1, 1.4])

    with col_img:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if heatmap is not None:
            tab_orig, tab_heat = st.tabs(["Image originale", "Carte d'attention"])
            with tab_orig:
                st.image(img, use_container_width=True)
            with tab_heat:
                overlay = overlay_heatmap(img, heatmap, alpha=0.5)
                st.image(overlay, use_container_width=True,
                         caption="Attention Rollout - zones analysees par le ViT")
                st.caption("Rouge = forte attention  |  Bleu = faible attention")
        else:
            st.image(img, use_container_width=True, caption=uploaded.name)
        if show_recon and recon is not None:
            st.image(recon, use_container_width=True, caption="Reconstruction AE")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_res:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Score d'anomalie (AE)**")
        if anom_model:
            st.plotly_chart(chart_anomaly_gauge(score_anom, anom_threshold),
                            use_container_width=True, config={"displayModeBar": False})
            status = "CAS ATYPIQUE - score au-dessus du seuil" if is_anomaly else "Dans la distribution normale"
            st.caption(f"{status}  (seuil = {anom_threshold:.5f})")
        else:
            st.warning(f"Modele anomalie : {anom_err}")
        st.markdown('</div>', unsafe_allow_html=True)

        if top_findings:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Pathologies detectees**")
            for name, prob, _ in top_findings[:5]:
                bar_color = "#da3633" if SEVERITY[name] >= 0.7 else "#d29922"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;margin:.25rem 0">'
                    f'<span>{LABEL_FR[name]}</span>'
                    f'<span style="color:{bar_color};font-weight:700">{prob:.1%}</span></div>',
                    unsafe_allow_html=True,
                )
                st.progress(float(prob))
            if len(top_findings) > 5:
                st.caption(f"+ {len(top_findings)-5} autre(s) pathologie(s) detectee(s)")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.success("Aucune pathologie detectee au-dessus du seuil.")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Probabilites - 14 pathologies")
    if clf_model:
        st.plotly_chart(chart_classification(probs, threshold),
                        use_container_width=True, config={"displayModeBar": False})
    else:
        st.warning(f"Modele classification : {clf_err}")

    if report_text.strip():
        st.markdown("### Analyse du compte-rendu")
        keywords = [LABEL_FR[l] for l in LABEL_NAMES
                    if l.lower() in report_text.lower() or LABEL_FR[l].lower() in report_text.lower()]
        col_kw, col_info = st.columns([1, 1])
        with col_kw:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if keywords:
                st.markdown(f"**Pathologies mentionnees :** {', '.join(keywords)}")
            else:
                st.info("Aucune pathologie reconnue dans le compte-rendu.")
            st.caption(f"{len(report_text.split())} mots")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_info:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.info("Modele multimodal entraine. AUC text_only = 0.9727 sur OpenI.")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    report = {
        "timestamp": datetime.now().isoformat(),
        "fichier": uploaded.name,
        "triage": {"niveau": level, "score_severite": round(sev_score, 4)},
        "anomalie": {
            "score": round(score_anom, 6),
            "seuil": round(anom_threshold, 6),
            "atypique": bool(is_anomaly),
        },
        "classification": {
            LABEL_NAMES[i]: round(float(probs[i]), 4) for i in range(len(LABEL_NAMES))
        },
        "pathologies_detectees": [
            {"nom": LABEL_FR[n], "probabilite": round(p, 4), "severite": round(s, 4)}
            for n, p, s in top_findings
        ],
        "compte_rendu": report_text.strip() or None,
        "modele_classification": artifact_key,
    }
    col_dl, col_metrics = st.columns([1, 2])
    with col_dl:
        st.download_button(
            "Telecharger le rapport JSON",
            data=json.dumps(report, ensure_ascii=False, indent=2),
            file_name=f"triage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("Pathologies detectees", len(top_findings))
        m2.metric("Score anomalie", f"{score_anom:.5f}")
        m3.metric("Modele", (artifact_key or "?").replace("model_", "").upper())


def _welcome():
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Classification")
        st.markdown("DeiT-tiny - Vision Transformer\n\nAUC macro = **0.7374** sur ChestMNIST (14 pathologies)")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Detection d'anomalies")
        st.markdown("Autoencoder convolutionnel\n\nSeuil au percentile 95 - taux anomalie test : **4.6%**")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_c:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Multimodal")
        st.markdown("TF-IDF + DenseNet fusion tardive\n\nAUC text_only = **0.9727** sur OpenI (3 818 paires)")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("### Chargez une radiographie pour demarrer l'analyse")


if __name__ == "__main__":
    main()
