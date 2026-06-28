import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{ROOT / 'mlflow.db'}"

st.set_page_config(page_title="Dashboard - TRI-AI", layout="wide")

st.markdown("""
<style>
  .stApp { background-color: #0d1117; color: #e6edf3; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:12px; padding:1rem 1.4rem; margin-bottom:1rem; }
  h1,h2,h3 { color:#58a6ff !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("# Dashboard - Resultats des modeles")
st.caption("Donnees chargees depuis MLflow (mlflow.db)")
st.divider()


@st.cache_data(ttl=60)
def load_mlflow_data():
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        data = {}
        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
            )
            data[exp.name] = runs
        return data, None
    except Exception as e:
        return {}, str(e)


data, err = load_mlflow_data()
if err:
    st.error(f"Erreur MLflow : {err}")
    st.stop()

import plotly.graph_objects as go
import pandas as pd

clf_key = next((k for k in data if "classification" in k), None)
if clf_key:
    st.markdown("## Classification (ChestMNIST)")
    runs = data[clf_key]
    if runs:
        rows = []
        for r in runs:
            m = r.data.metrics
            rows.append({
                "Modele":    r.info.run_name or "-",
                "AUC macro": round(m.get("test_auc_macro", 0), 4),
                "F1 macro":  round(m.get("test_f1_macro", 0), 4),
                "F1 micro":  round(m.get("test_f1_micro", 0), 4),
                "MCC macro": round(m.get("test_mcc_macro", 0), 4),
                "Bal. Acc.": round(m.get("test_balanced_accuracy", 0), 4),
                "Meilleur":  "oui" if r.data.tags.get("best_model") == "true" else "",
            })

        df = pd.DataFrame(rows).sort_values("AUC macro", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)

        metrics_to_show = ["AUC macro", "F1 macro", "MCC macro", "Bal. Acc."]
        colors = ["#388bfd", "#3fb950", "#d29922", "#a371f7"]
        fig = go.Figure()
        for i, metric in enumerate(metrics_to_show):
            fig.add_trace(go.Bar(
                name=metric,
                x=df["Modele"], y=df[metric],
                marker_color=colors[i],
                text=[f"{v:.3f}" for v in df[metric]],
                textposition="outside",
            ))
        fig.update_layout(
            barmode="group",
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            font_color="#e6edf3", height=360,
            legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis=dict(range=[0, 1.05], gridcolor="#21262d"),
            xaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig, use_container_width=True)

        best_run = next((r for r in runs if r.data.tags.get("best_model") == "true"), runs[0])
        auc_per_class = {
            k.replace("test_auc_", ""): v
            for k, v in best_run.data.metrics.items()
            if k.startswith("test_auc_") and k != "test_auc_macro"
        }
        if auc_per_class:
            st.markdown(f"**AUC par classe - {best_run.info.run_name}**")
            sorted_auc = dict(sorted(auc_per_class.items(), key=lambda x: x[1]))
            fig2 = go.Figure(go.Bar(
                x=list(sorted_auc.values()),
                y=list(sorted_auc.keys()),
                orientation="h",
                marker_color=["#da3633" if v < 0.65 else "#d29922" if v < 0.75 else "#238636"
                              for v in sorted_auc.values()],
                text=[f"{v:.3f}" for v in sorted_auc.values()],
                textposition="outside",
                textfont=dict(color="white"),
            ))
            fig2.add_vline(x=0.7, line_dash="dash", line_color="#58a6ff",
                           annotation_text="AUC=0.70", annotation_font_color="#58a6ff")
            fig2.update_layout(
                paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                font_color="#e6edf3", height=400,
                xaxis=dict(range=[0, 1.1], gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
                margin=dict(l=10, r=80, t=10, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

st.divider()

anom_key = next((k for k in data if "anomaly" in k), None)
if anom_key:
    st.markdown("## Detection d'anomalies")
    runs_a = data[anom_key]
    if runs_a:
        r = runs_a[0]
        m = r.data.metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Seuil anomalie", f"{m.get('anomaly_threshold', 0):.5f}")
        c2.metric("Taux anomalie test", f"{m.get('test_anomaly_rate', 0):.1%}")
        c3.metric("Images normales train", str(int(r.data.params.get("n_normal_train", 0))))
        c4.metric("Modele", r.data.tags.get("model_type", "ae_conv").upper())

st.divider()

mm_key = next((k for k in data if "multimodal" in k), None)
if mm_key:
    st.markdown("## Fusion multimodale (OpenI)")
    runs_m = data[mm_key]
    if runs_m:
        rows_m = []
        for r in runs_m:
            m = r.data.metrics
            rows_m.append({
                "Mode":     r.data.tags.get("fusion_mode", r.info.run_name or "?"),
                "AUC test": round(m.get("test_auc_macro", 0), 4),
                "F1 macro": round(m.get("test_f1_macro", 0), 4),
                "MCC":      round(m.get("test_mcc_macro", 0), 4),
                "Bal. Acc": round(m.get("test_balanced_accuracy", 0), 4),
                "Meilleur": "oui" if r.data.tags.get("best_model") == "true" else "",
            })
        df_m = pd.DataFrame(rows_m).sort_values("AUC test", ascending=False)
        st.dataframe(df_m, use_container_width=True, hide_index=True)

        fig_m = go.Figure(go.Bar(
            x=df_m["Mode"], y=df_m["AUC test"],
            marker_color=["#d29922" if b == "oui" else "#388bfd" for b in df_m["Meilleur"]],
            text=[f"{v:.4f}" for v in df_m["AUC test"]],
            textposition="outside", textfont=dict(color="white"),
        ))
        fig_m.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            font_color="#e6edf3", height=280,
            yaxis=dict(range=[0, 1.1], gridcolor="#21262d"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_m, use_container_width=True)
        st.info("Le mode text_only (TF-IDF) surpasse la fusion : les comptes-rendus OpenI "
                "contiennent les diagnostics explicitement.")
