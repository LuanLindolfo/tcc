import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import find_col, load_demografico

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
    c = find_col(df, ["pop_total", "populacao_total", "total", "populacao", "pessoas"])
    pop = int(df[c].sum()) if c else "N/D"
    st.metric("👥 População Total", f"{pop:,}".replace(",", ".") if isinstance(pop, int) else pop)
with col2:
    c = find_col(df, ["indice_envelhecimento", "ie", "envelhecimento"])
    ie = float(df[c].mean()) if c else None
    st.metric("👴 Índice de Envelhecimento", f"{ie:.1f}" if ie else "N/D")
with col3:
    c = find_col(df, ["razao_sexo", "razao_de_sexo", "sex_ratio"])
    rs = float(df[c].mean()) if c else None
    st.metric("⚖️ Razão de Sexo", f"{rs:.1f} H/100M" if rs else "N/D")
with col4:
    st.metric("🗺️ Linhas carregadas", len(df))

st.divider()

# ── Pirâmide Etária ───────────────────────────────────────────────────────────
st.subheader("🔺 Pirâmide Etária")

# Tenta encontrar colunas de sexo desagregado
col_masc = find_col(df, ["homens", "masculino", "masc", "pop_masc", "h"])
col_fem  = find_col(df, ["mulheres", "feminino", "fem", "pop_fem", "m"])
col_faixa = find_col(df, ["faixa", "faixa_de_idade", "faixa_etaria", "grupo_de_idade", "idade"])

if col_masc and col_fem and col_faixa:
    df_pir = df[[col_faixa, col_masc, col_fem]].copy()
    df_pir.columns = ["Faixa", "Masculino", "Feminino"]
    df_pir = df_pir.dropna()
    fig_piramide = go.Figure()
    fig_piramide.add_trace(go.Bar(
        y=df_pir["Faixa"], x=-df_pir["Masculino"].astype(float),
        name="Masculino", orientation="h", marker_color="#2196F3",
    ))
    fig_piramide.add_trace(go.Bar(
        y=df_pir["Faixa"], x=df_pir["Feminino"].astype(float),
        name="Feminino", orientation="h", marker_color="#E91E63",
    ))
    fig_piramide.update_layout(
        barmode="overlay", title="Pirâmide Etária — Castanhal 2022",
        xaxis_title="População", yaxis_title="Faixa Etária", height=400,
    )
    st.plotly_chart(fig_piramide, use_container_width=True)
elif col_faixa:
    # Fallback: qualquer coluna numérica como total
    col_val = find_col(df, ["total", "populacao", "pessoas", "valor"])
    if col_val:
        fig_piramide = px.bar(
            df.dropna(subset=[col_faixa, col_val]),
            x=col_faixa, y=col_val,
            title="Distribuição por Faixa Etária — Castanhal 2022",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig_piramide, use_container_width=True)
    else:
        st.info("Colunas de valor não encontradas nos dados de pirâmide etária.")
else:
    st.info("Colunas de faixa etária não encontradas. Execute o pipeline Colab para gerar os dados.")

with st.expander("🔍 Diagnóstico — colunas disponíveis no parquet demográfico"):
    st.write(list(df.columns))

# ── Distribuição Étnico-Racial ────────────────────────────────────────────────
st.subheader("🌈 Distribuição Étnico-Racial")
st.caption(
    "⚠️ Dados de distribuição étnico-racial estão sujeitos a limitações dos questionários "
    "censitários e possíveis vieses de autodeclaração — interprete com cautela."
)
raca_candidatos = {
    "Branca":   ["branca", "pct_branca", "cor_branca"],
    "Preta":    ["preta", "pct_preta", "cor_preta"],
    "Parda":    ["parda", "pct_parda", "cor_parda"],
    "Indígena": ["indigena", "pct_indigena", "cor_indigena"],
    "Amarela":  ["amarela", "pct_amarela", "cor_amarela"],
}
raca_data = {}
for label, cands in raca_candidatos.items():
    c = find_col(df, cands)
    if c:
        raca_data[label] = float(df[c].mean())

if raca_data:
    df_raca = pd.DataFrame(list(raca_data.items()), columns=["Cor/Raça", "% Média"])
    fig_raca = px.pie(
        df_raca, names="Cor/Raça", values="% Média",
        title="Distribuição por Cor/Raça — Média dos Setores",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    st.plotly_chart(fig_raca, use_container_width=True)
else:
    st.info("Dados étnico-raciais não disponíveis nos dados processados.")

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
