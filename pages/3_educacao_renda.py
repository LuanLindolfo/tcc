import streamlit as st
import plotly.express as px
import pandas as pd

from utils.data_loader import find_col, load_educacao, load_trabalho_renda

st.set_page_config(page_title="Educação & Renda — Castanhal", layout="wide")

st.title("📚 Educação & Trabalho/Renda")
st.markdown("Indicadores educacionais e econômicos de **Castanhal – PA** (Censo IBGE 2022)")
st.divider()

with st.spinner("Carregando dados de educação e renda..."):
    df_edu = load_educacao()
    df_ren = load_trabalho_renda()

if df_edu.empty and df_ren.empty:
    st.error(
        "❌ Dados de educação e renda não disponíveis. Execute o pipeline no Colab para gerar os dados.",
        icon="❌",
    )
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    c = find_col(df_edu, ["taxa_analfabetismo", "analfabetismo", "analf", "taxa_de_analfabetismo"])
    analf = float(df_edu[c].mean()) if c else None
    st.metric("📖 Taxa de Analfabetismo", f"{analf:.1f}%" if analf else "N/D")
with col2:
    c = find_col(df_ren, ["renda_media_per_capita", "rendimento", "renda", "rendimento_domiciliar"])
    renda = float(df_ren[c].mean()) if c else None
    st.metric("💰 Renda Média per Capita", f"R$ {renda:,.0f}".replace(",", ".") if renda else "N/D")
with col3:
    freq = df_edu["pct_freq_escolar_criancas"].mean() if "pct_freq_escolar_criancas" in df_edu.columns else None
    st.metric("🏫 Freq. Escolar (6–14 anos)", f"{freq:.1f}%" if freq else "N/D")
with col4:
    pea = df_ren["taxa_atividade_pea"].mean() if "taxa_atividade_pea" in df_ren.columns else None
    st.metric("👷 Taxa de Atividade (PEA)", f"{pea:.1f}%" if pea else "N/D")

st.divider()

# ── Nível de Escolaridade ──────────────────────────────────────────────────────
st.subheader("🎓 Distribuição por Nível de Instrução")
esc_cols = {
    "Fund. Incompleto": "pct_fundamental_incompleto",
    "Fund. Completo": "pct_fundamental_completo",
    "Médio Incompleto": "pct_medio_incompleto",
    "Médio Completo": "pct_medio_completo",
    "Superior Incompleto": "pct_superior_incompleto",
    "Superior Completo": "pct_superior_completo",
}
esc_data = {k: df_edu[v].mean() for k, v in esc_cols.items() if v in df_edu.columns}
if esc_data:
    df_esc = pd.DataFrame(list(esc_data.items()), columns=["Nível", "% Médio"])
    df_esc = df_esc.sort_values("% Médio", ascending=True)
    fig_esc = px.funnel(
        df_esc, x="% Médio", y="Nível",
        title="Funil de Escolaridade — Castanhal 2022",
        color_discrete_sequence=["#5C6BC0"],
    )
    st.plotly_chart(fig_esc, use_container_width=True)
else:
    st.info("Dados de nível de instrução não disponíveis.")

# ── Scatter Renda × Escolaridade ──────────────────────────────────────────────
if not df_edu.empty and not df_ren.empty and "setor_id" in df_edu.columns and "setor_id" in df_ren.columns:
    st.subheader("📈 Relação entre Escolaridade e Renda per Capita")
    df_merge = df_edu.merge(df_ren, on="setor_id", how="inner")
    if "escolaridade_media_anos" in df_merge.columns and "renda_media_per_capita" in df_merge.columns:
        fig_scatter = px.scatter(
            df_merge, x="escolaridade_media_anos", y="renda_media_per_capita",
            title="Escolaridade Média vs. Renda Domiciliar per Capita por Setor",
            labels={"escolaridade_media_anos": "Anos médios de estudo",
                    "renda_media_per_capita": "Renda média per capita (R$)"},
            trendline="ols",
            color_discrete_sequence=["#26A69A"],
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ── Distribuição de Renda ──────────────────────────────────────────────────────
if "renda_media_per_capita" in df_ren.columns:
    st.subheader("💵 Distribuição de Renda Domiciliar per Capita")
    fig_renda = px.histogram(
        df_ren, x="renda_media_per_capita", nbins=25,
        title="Distribuição de Renda per Capita por Setor Censitário",
        labels={"renda_media_per_capita": "Renda (R$)"},
        color_discrete_sequence=["#FFA726"],
    )
    st.plotly_chart(fig_renda, use_container_width=True)

# ── Índice de Gini (DIFERIDO) ─────────────────────────────────────────────────
st.subheader("📊 Desigualdade de Renda (Índice de Gini)")
st.info(
    "⏳ **Índice de Gini**: Esta análise será disponibilizada em versão futura, "
    "após a validação dos microdados necessários para o cálculo. "
    "O campo está reservado no modelo de dados.",
    icon="⏳",
)

# ── PEA por Setor Econômico ────────────────────────────────────────────────────
st.subheader("🏭 Distribuição da PEA por Setor Econômico")
pea_cols = {
    "🌾 Primário": "pct_setor_primario",
    "🏭 Secundário": "pct_setor_secundario",
    "🛒 Terciário": "pct_setor_terciario",
}
pea_data = {k: df_ren[v].mean() for k, v in pea_cols.items() if v in df_ren.columns}
if pea_data:
    df_pea = pd.DataFrame(list(pea_data.items()), columns=["Setor", "% Médio"])
    fig_pea = px.bar(
        df_pea, x="Setor", y="% Médio",
        title="Distribuição da PEA por Setor de Atividade Econômica",
        color="Setor",
        color_discrete_sequence=["#8D6E63", "#78909C", "#42A5F5"],
        text_auto=".1f",
    )
    fig_pea.update_layout(showlegend=False, yaxis_range=[0, 100])
    st.plotly_chart(fig_pea, use_container_width=True)
else:
    st.info("Dados de distribuição da PEA por setor não disponíveis.")

with st.expander("📋 Ver dados brutos — Educação"):
    st.dataframe(df_edu, use_container_width=True)
with st.expander("📋 Ver dados brutos — Trabalho & Renda"):
    st.dataframe(df_ren, use_container_width=True)
