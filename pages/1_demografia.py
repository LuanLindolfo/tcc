import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_demografico

st.set_page_config(page_title="Dinâmica Populacional — Castanhal", layout="wide")

st.title("📊 Dinâmica Populacional")
st.markdown("Análise demográfica do município de **Castanhal – PA** (Censo IBGE 2022)")
st.divider()

with st.spinner("Carregando dados demográficos..."):
    df = load_demografico()

if df.empty:
    st.error(
        "❌ Dados demográficos não disponíveis. Execute o pipeline no Colab para gerar os dados.",
        icon="❌",
    )
    st.stop()

# ── KPIs ─────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    pop = int(df["pop_total"].sum()) if "pop_total" in df.columns else "N/D"
    st.metric("👥 População Total", f"{pop:,}".replace(",", ".") if isinstance(pop, int) else pop)
with col2:
    ie = df["indice_envelhecimento"].mean() if "indice_envelhecimento" in df.columns else None
    st.metric("👴 Índice de Envelhecimento", f"{ie:.1f}" if ie else "N/D")
with col3:
    rs = df["razao_sexo"].mean() if "razao_sexo" in df.columns else None
    st.metric("⚖️ Razão de Sexo", f"{rs:.1f} H/100M" if rs else "N/D")
with col4:
    n_setores = len(df)
    st.metric("🗺️ Setores Censitários", n_setores)

st.divider()

# ── Pirâmide Etária ───────────────────────────────────────────────────────────
st.subheader("🔺 Pirâmide Etária")
faixas = []
for col_masc, col_fem, faixa in [
    ("pop_masc_0_14", "pop_fem_0_14", "0–14 anos"),
    ("pop_masc_15_64", "pop_fem_15_64", "15–64 anos"),
    ("pop_masc_65_mais", "pop_fem_65_mais", "65+ anos"),
]:
    if col_masc in df.columns and col_fem in df.columns:
        faixas.append({"Faixa": faixa, "Masculino": df[col_masc].sum(), "Feminino": df[col_fem].sum()})

if faixas:
    import pandas as pd
    df_piramide = pd.DataFrame(faixas)
    fig_piramide = go.Figure()
    fig_piramide.add_trace(go.Bar(
        y=df_piramide["Faixa"], x=-df_piramide["Masculino"],
        name="Masculino", orientation="h", marker_color="#2196F3",
    ))
    fig_piramide.add_trace(go.Bar(
        y=df_piramide["Faixa"], x=df_piramide["Feminino"],
        name="Feminino", orientation="h", marker_color="#E91E63",
    ))
    fig_piramide.update_layout(
        barmode="overlay", title="Pirâmide Etária — Castanhal 2022",
        xaxis_title="População", yaxis_title="Faixa Etária",
        height=350,
    )
    st.plotly_chart(fig_piramide, use_container_width=True)
elif all(c in df.columns for c in ["pop_0_14", "pop_15_64", "pop_65_mais"]):
    # Fallback: barras simples sem sex-split
    import pandas as pd
    df_faixas = pd.DataFrame({
        "Faixa Etária": ["0–14 anos", "15–64 anos", "65+ anos"],
        "População": [df["pop_0_14"].sum(), df["pop_15_64"].sum(), df["pop_65_mais"].sum()],
    })
    fig_piramide = px.bar(
        df_faixas, x="Faixa Etária", y="População",
        title="Distribuição por Faixa Etária — Castanhal 2022",
        color="Faixa Etária", color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    st.plotly_chart(fig_piramide, use_container_width=True)
else:
    st.info("Colunas de faixa etária não encontradas nos dados.")

# ── Distribuição Étnico-Racial ────────────────────────────────────────────────
st.subheader("🌈 Distribuição Étnico-Racial")
raca_cols = {
    "Branca": "pct_branca", "Preta": "pct_preta", "Parda": "pct_parda",
    "Indígena": "pct_indigena", "Amarela": "pct_amarela",
}
raca_data = {k: df[v].mean() for k, v in raca_cols.items() if v in df.columns}
if raca_data:
    import pandas as pd
    df_raca = pd.DataFrame(list(raca_data.items()), columns=["Cor/Raça", "% Média"])
    fig_raca = px.pie(
        df_raca, names="Cor/Raça", values="% Média",
        title="Distribuição por Cor/Raça — Média dos Setores",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    st.plotly_chart(fig_raca, use_container_width=True)
else:
    st.info("Dados étnico-raciais não disponíveis.")

# ── Migração ──────────────────────────────────────────────────────────────────
if "pct_naturais_castanhal" in df.columns or "pct_migrantes" in df.columns:
    st.subheader("🚶 Perfil Migratório")
    col_a, col_b = st.columns(2)
    if "pct_naturais_castanhal" in df.columns:
        col_a.metric("🏠 Naturais de Castanhal", f"{df['pct_naturais_castanhal'].mean():.1f}%")
    if "pct_migrantes" in df.columns:
        col_b.metric("✈️ Migrantes (fora do Pará)", f"{df['pct_migrantes'].mean():.1f}%")

# ── Tabela resumo ─────────────────────────────────────────────────────────────
with st.expander("📋 Ver dados brutos por setor censitário"):
    st.dataframe(df, use_container_width=True)
