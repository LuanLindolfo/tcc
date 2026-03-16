import streamlit as st
import plotly.express as px
import pandas as pd

from utils.data_loader import find_col, load_domicilios

st.set_page_config(page_title="Diagnóstico Habitacional — Castanhal", layout="wide")

st.title("🏠 Diagnóstico Habitacional")
st.markdown("Condições de moradia e infraestrutura dos domicílios de **Castanhal – PA** (Censo 2022)")
st.divider()

with st.spinner("Carregando dados habitacionais..."):
    df = load_domicilios()

if df.empty:
    st.error(
        "❌ Dados habitacionais não disponíveis. Execute o pipeline no Colab para gerar os dados.",
        icon="❌",
    )
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    c = find_col(df, ["total_domicilios", "domicilios", "total", "quantidade"])
    total_dom = int(df[c].sum()) if c else "N/D"
    st.metric("🏘️ Total de Domicílios", f"{total_dom:,}".replace(",", ".") if isinstance(total_dom, int) else total_dom)
with col2:
    c = find_col(df, ["media_moradores", "media_de_moradores", "moradores", "numero_de_moradores"])
    media_mor = float(df[c].mean()) if c else None
    st.metric("👨‍👩‍👧 Média de Moradores", f"{media_mor:.1f}" if media_mor else "N/D")
with col3:
    c = find_col(df, ["iah", "indice_adequacao_habitacional", "adequacao"])
    iah_medio = float(df[c].mean()) if c else None
    st.metric("🏗️ IAH Médio", f"{iah_medio:.2f}" if iah_medio else "N/D", help="Índice de Adequação Habitacional (0-1)")
with col4:
    cands_san = ["agua", "esgoto", "lixo", "energia", "saneamento"]
    cols_san = [find_col(df, [k]) for k in cands_san]
    available = [c for c in cols_san if c is not None]
    pct_saneamento = float(df[available].mean().mean()) if available else None
    st.metric("🚰 Saneamento Básico Médio", f"{pct_saneamento:.1f}%" if pct_saneamento else "N/D")

st.divider()

# ── Mapa de IAH ───────────────────────────────────────────────────────────────
if "iah" in df.columns:
    st.subheader("🗺️ Distribuição do IAH por Setor")
    st.caption("Índice de Adequação Habitacional: quanto mais próximo de 1, melhor a infraestrutura.")
    fig_iah = px.histogram(
        df, x="iah", nbins=20,
        title="Distribuição do Índice de Adequação Habitacional (IAH)",
        labels={"iah": "IAH (0 = inadequado, 1 = adequado)"},
        color_discrete_sequence=["#FF7043"],
    )
    fig_iah.add_vline(x=0.5, line_dash="dash", line_color="red",
                      annotation_text="Limiar crítico (0,5)")
    st.plotly_chart(fig_iah, use_container_width=True)

# ── Saneamento Básico ─────────────────────────────────────────────────────────
st.subheader("🚿 Acesso a Serviços de Saneamento Básico")
san_data = {
    "💧 Água Encanada":    ["pct_agua_encanada", "agua_encanada", "agua"],
    "🪣 Esgoto Sanitário": ["pct_esgoto", "esgoto", "esgoto_sanitario"],
    "🗑️ Coleta de Lixo":  ["pct_coleta_lixo", "coleta_lixo", "lixo"],
    "⚡ Energia Elétrica": ["pct_energia_eletrica", "energia_eletrica", "energia"],
}
san_medias = {}
for label, cands in san_data.items():
    c = find_col(df, cands)
    if c:
        san_medias[label] = float(df[c].mean())
if san_medias:
    df_san = pd.DataFrame(list(san_medias.items()), columns=["Serviço", "% Média dos Setores"])
    fig_san = px.bar(
        df_san, x="Serviço", y="% Média dos Setores",
        title="Cobertura Média de Serviços de Saneamento — Castanhal 2022",
        color="Serviço",
        color_discrete_sequence=px.colors.qualitative.Pastel2,
        text_auto=".1f",
    )
    fig_san.update_layout(yaxis_range=[0, 100], showlegend=False)
    st.plotly_chart(fig_san, use_container_width=True)

# ── Tipos de Domicílio ────────────────────────────────────────────────────────
st.subheader("🏗️ Tipos de Domicílio")
tipo_data = {
    "🏠 Casas": "pct_casas",
    "🏢 Apartamentos": "pct_apartamentos",
    "🛏️ Cômodos": "pct_comodos",
}
tipo_medias = {k: df[v].mean() for k, v in tipo_data.items() if v in df.columns}
if tipo_medias:
    df_tipo = pd.DataFrame(list(tipo_medias.items()), columns=["Tipo", "% Médio"])
    fig_tipo = px.pie(
        df_tipo, names="Tipo", values="% Médio",
        title="Distribuição de Tipos de Domicílio",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hole=0.4,
    )
    st.plotly_chart(fig_tipo, use_container_width=True)

# ── Posse e Ocupação ──────────────────────────────────────────────────────────
st.subheader("📋 Posse e Condição de Ocupação")
posse_data = {
    "🏠 Próprio": "pct_proprio",
    "🔑 Alugado": "pct_alugado",
    "🎁 Cedido": "pct_cedido",
}
posse_medias = {k: df[v].mean() for k, v in posse_data.items() if v in df.columns}
if posse_medias:
    df_posse = pd.DataFrame(list(posse_medias.items()), columns=["Condição", "% Médio"])
    fig_posse = px.bar(
        df_posse, x="Condição", y="% Médio",
        title="Condição de Ocupação dos Domicílios",
        color="Condição",
        color_discrete_sequence=["#42A5F5", "#EF5350", "#66BB6A"],
        text_auto=".1f",
    )
    fig_posse.update_layout(showlegend=False, yaxis_range=[0, 100])
    st.plotly_chart(fig_posse, use_container_width=True)

# ── Tabela ────────────────────────────────────────────────────────────────────
with st.expander("📋 Ver dados brutos por setor censitário"):
    st.dataframe(df, use_container_width=True)
