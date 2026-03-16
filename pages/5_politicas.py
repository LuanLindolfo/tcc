import streamlit as st
import plotly.express as px
import pandas as pd

from utils.data_loader import load_politicas, load_features_compostas

st.set_page_config(page_title="Políticas Públicas — Castanhal", layout="wide")

st.title("🏛️ Políticas Públicas de Castanhal")
st.markdown(
    "Recomendações baseadas nos dados do Censo 2022 e nos resultados dos modelos de ML. "
    "As análises são de **apoio à decisão** — não substituem o processo político-administrativo."
)
st.divider()

with st.spinner("Carregando dados de políticas..."):
    politicas = load_politicas()
    df = load_features_compostas()

if not politicas:
    st.warning(
        "⚠️ Nenhuma política pública cadastrada. Execute o notebook Colab para gerar os artefatos.",
        icon="⚠️",
    )
    st.stop()

# ── Seletor de área temática ──────────────────────────────────────────────────
areas = [p["politica"] for p in politicas]
area_selecionada = st.selectbox("🔍 Selecione uma área temática:", areas)
politica = next((p for p in politicas if p["politica"] == area_selecionada), None)

if not politica:
    st.error("Política não encontrada.")
    st.stop()

st.divider()

# ── Card da política ──────────────────────────────────────────────────────────
col_info, col_modelo = st.columns([2, 1])
with col_info:
    st.subheader(f"🏛️ {politica['politica']}")
    st.markdown(politica["descricao"])
with col_modelo:
    st.metric("📊 Indicador-chave", politica.get("indicador_chave", "N/D").replace("_", " ").title())
    st.metric("🤖 Modelo ML", politica.get("modelo_relacionado", "N/D").replace("_", " ").title())

st.info(f"💡 **Recomendação de Ação:** {politica.get('recomendacao', 'N/D')}", icon="💡")
st.divider()

# ── Análise dos setores prioritários ─────────────────────────────────────────
if df.empty:
    st.warning("Dados dos setores não disponíveis. Execute o pipeline Colab.")
else:
    indicador = politica.get("indicador_chave")
    limiar = politica.get("limiar")

    if indicador and indicador in df.columns:
        st.subheader(f"📋 Top 10 Setores Prioritários — {indicador.replace('_', ' ').title()}")

        if limiar is not None:
            setores_crit = df[df[indicador] < limiar].copy()
            st.caption(f"Setores com `{indicador}` abaixo de **{limiar}** (limiar crítico)")
        else:
            q3 = df[indicador].quantile(0.75)
            setores_crit = df[df[indicador] >= q3].copy()
            st.caption(f"Setores acima do 3º quartil de `{indicador}` (≥ {q3:.2f})")

        if setores_crit.empty:
            st.success("✅ Nenhum setor em situação crítica para este indicador.")
        else:
            colunas_exibir = ["setor_id", indicador]
            if "bairro" in df.columns:
                colunas_exibir.insert(1, "bairro")
            exibir = [c for c in colunas_exibir if c in setores_crit.columns]
            top10 = setores_crit.sort_values(indicador).head(10)
            st.dataframe(top10[exibir], use_container_width=True)

            fig_bar = px.bar(
                top10, x="setor_id" if "setor_id" in top10.columns else top10.index.astype(str),
                y=indicador,
                title=f"10 Setores mais críticos — {indicador.replace('_', ' ').title()}",
                color=indicador,
                color_continuous_scale="RdYlGn_r",
                labels={"setor_id": "Setor Censitário", indicador: indicador.replace("_", " ").title()},
            )
            if limiar:
                fig_bar.add_hline(y=limiar, line_dash="dash", line_color="red",
                                  annotation_text=f"Limiar crítico ({limiar})")
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info(f"Indicador `{indicador}` não encontrado nos dados processados.")

# ── Visão geral de todas as políticas ─────────────────────────────────────────
with st.expander("📋 Ver todas as políticas cadastradas"):
    df_pol = pd.DataFrame([{
        "Política": p["politica"],
        "Área": p["area"],
        "Indicador-chave": p["indicador_chave"],
        "Modelo ML": p["modelo_relacionado"],
    } for p in politicas])
    st.dataframe(df_pol, use_container_width=True)
