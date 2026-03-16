import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from utils.data_loader import load_features_compostas, load_ml_metricas

st.set_page_config(page_title="Machine Learning — Castanhal", layout="wide")

st.title("🤖 Resultados de Machine Learning")
st.markdown("Modelos treinados sobre o Censo 2022 de **Castanhal – PA** com transparência algorítmica total.")
st.divider()

with st.spinner("Carregando dados e métricas de ML..."):
    df = load_features_compostas()
    met_clf = load_ml_metricas("classificacao")
    met_reg = load_ml_metricas("regressao")

sem_artefatos = df.empty and not met_clf and not met_reg
if sem_artefatos:
    st.warning(
        "⚠️ Modelos ainda não treinados. Execute o notebook Colab para gerar os artefatos de ML.",
        icon="⚠️",
    )
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "🏷️ Classificação — IVS",
    "📏 Regressão — IAH",
    "🔵 Clustering — Ocupação",
])

# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Vulnerabilidade Socioeconômica (IVS)")
    st.caption("Modelo: Random Forest Classifier | Target: baixa / media / alta")

    if met_clf:
        c1, c2, c3 = st.columns(3)
        c1.metric("🎯 Acurácia", f"{met_clf.get('acuracia', 0):.2%}")
        c2.metric("📊 F1 Macro", f"{met_clf.get('f1_macro', 0):.2%}")
        c3.metric("🗃️ Classes", ", ".join(met_clf.get("classes", [])))
        st.divider()

        # Matriz de Confusão
        mc = met_clf.get("matriz_confusao")
        classes = met_clf.get("classes", [])
        if mc and classes:
            st.markdown("**Matriz de Confusão**")
            fig_mc = px.imshow(
                mc, x=classes, y=classes,
                color_continuous_scale="Blues",
                labels=dict(x="Predito", y="Real", color="Contagem"),
                text_auto=True,
                title="Matriz de Confusão — Random Forest IVS",
            )
            st.plotly_chart(fig_mc, use_container_width=True)

        # Feature Importance
        fi = met_clf.get("feature_importances", {})
        if fi and isinstance(fi, dict):
            df_fi = pd.DataFrame(list(fi.items()), columns=["Feature", "Importância"])
            df_fi["Importância"] = pd.to_numeric(df_fi["Importância"], errors="coerce").fillna(0)
            df_fi = df_fi.sort_values("Importância")
            fig_fi = px.bar(
                df_fi, x="Importância", y="Feature", orientation="h",
                title="Top Features — Importância para Classificação IVS",
                color="Importância", color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Métricas de classificação não disponíveis.")

    # Distribuição IVS nos dados
    if not df.empty and "ivs_label" in df.columns:
        st.markdown("**Distribuição dos Setores por Nível de Vulnerabilidade**")
        fig_ivs = px.pie(
            df, names="ivs_label",
            title="Distribuição de Vulnerabilidade Socioeconômica — Setores de Castanhal",
            color="ivs_label",
            color_discrete_map={"baixa": "#66BB6A", "media": "#FFA726", "alta": "#EF5350"},
        )
        st.plotly_chart(fig_ivs, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Infraestrutura Urbana — IAH (Índice de Adequação Habitacional)")
    st.caption("Modelo: XGBoost Regressor | Target: IAH (0–1)")

    if met_reg:
        c1, c2, c3 = st.columns(3)
        c1.metric("📈 R²", f"{met_reg.get('r2', 0):.4f}")
        c2.metric("📉 RMSE", f"{met_reg.get('rmse', 0):.4f}")
        c3.metric("📐 MAE", f"{met_reg.get('mae', 0):.4f}")
        st.divider()

        y_test = met_reg.get("y_test", [])
        y_pred = met_reg.get("y_pred", [])
        if y_test and y_pred:
            df_pred = pd.DataFrame({"IAH Real": y_test, "IAH Predito": y_pred})
            fig_pred = px.scatter(
                df_pred, x="IAH Real", y="IAH Predito",
                title="IAH Real vs. IAH Predito — XGBoost",
                trendline="ols",
                color_discrete_sequence=["#7E57C2"],
            )
            fig_pred.add_shape(
                type="line", x0=0, y0=0, x1=1, y1=1,
                line=dict(dash="dash", color="gray"),
            )
            st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.info("Métricas de regressão não disponíveis.")

    if not df.empty and "iah" in df.columns:
        fig_iah_dist = px.histogram(
            df, x="iah", nbins=20, color="ivs_label" if "ivs_label" in df.columns else None,
            title="Distribuição do IAH por Nível de Vulnerabilidade",
            labels={"iah": "IAH (0 = inadequado, 1 = adequado)"},
            color_discrete_map={"baixa": "#66BB6A", "media": "#FFA726", "alta": "#EF5350"},
        )
        st.plotly_chart(fig_iah_dist, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Perfil de Ocupação por Cluster — K-Means")
    st.caption("Segmentação dos setores pelo perfil econômico (primário, secundário, terciário + renda)")

    if not df.empty and "cluster_ocupacao" in df.columns:
        clusters_disponiveis = sorted(df["cluster_ocupacao"].unique().tolist())
        cluster_labels = {c: f"Cluster {c}" for c in clusters_disponiveis}

        # Filtro interativo — T016 (US2 Acceptance Scenario 2)
        col_filtro, col_info = st.columns([1, 3])
        with col_filtro:
            opcoes = ["Todos"] + [cluster_labels[c] for c in clusters_disponiveis]
            filtro = st.selectbox("🔍 Filtrar por Cluster", opcoes, key="cluster_select")
        with col_info:
            if filtro != "Todos":
                cl_num = int(filtro.split()[-1])
                n_setores_cluster = int((df["cluster_ocupacao"] == cl_num).sum())
                st.info(f"Cluster {cl_num}: **{n_setores_cluster}** setores")

        df_cl = df.copy()
        if filtro != "Todos":
            cl_num = int(filtro.split()[-1])
            df_cl = df_cl[df_cl["cluster_ocupacao"] == cl_num]

        feat_cluster = [c for c in df_cl.select_dtypes(include="number").columns
                        if c not in ["cluster_ocupacao", "iah"] and
                        any(x in c for x in ["setor", "renda", "rendimento", "primario",
                                              "secundario", "terciario", "ocupacao"])]
        if not feat_cluster:
            feat_cluster = [c for c in df_cl.select_dtypes(include="number").columns
                            if c not in ["cluster_ocupacao"]]

        if len(feat_cluster) >= 2:
            fig_cl = px.scatter(
                df_cl, x=feat_cluster[0], y=feat_cluster[min(1, len(feat_cluster)-1)],
                color=df_cl["cluster_ocupacao"].astype(str),
                title=f"Clusters de Perfil de Ocupação — {filtro}",
                labels={feat_cluster[0]: feat_cluster[0].replace("_", " ").title(),
                        feat_cluster[1]: feat_cluster[1].replace("_", " ").title(),
                        "color": "Cluster"},
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            st.plotly_chart(fig_cl, use_container_width=True)

        st.markdown("**Distribuição de Setores por Cluster**")
        contagem = df_cl["cluster_ocupacao"].value_counts().reset_index()
        contagem.columns = ["Cluster", "Setores"]
        pct = (contagem["Setores"] / len(df) * 100).round(1)
        contagem["% do Total"] = pct.astype(str) + "%"
        st.dataframe(contagem, use_container_width=True)
    else:
        st.info("Dados de clustering não disponíveis. Execute o pipeline Colab.")

with st.expander("📋 Ver features completas por setor"):
    if not df.empty:
        st.dataframe(df, use_container_width=True)
