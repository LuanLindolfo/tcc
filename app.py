# -*- coding: utf-8 -*-
"""
app.py — Painel Streamlit TCC (coringa multi-município)
Edite apenas o bloco CONFIGURAÇÃO para adicionar novas cidades.
"""

from __future__ import annotations

import os
import re
import glob

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║              BLOCO DE CONFIGURAÇÃO — EDITE APENAS AQUI                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Título geral do painel
TITULO_PAINEL = "Censo IBGE — Projeções Municipais"

# Cada entrada é um município.
# "pasta" aponta para a pasta de dados dentro do repositório.
# "arquivo" é o nome do CSV exportado pelo notebook (seção 7).
# "nome"    é o rótulo exibido na aba.
MUNICIPIOS = [
    {
        "nome":    "Castanhal",
        "pasta":   "data",
        "arquivo": "castanhal_indicadores.csv",
    },
    {
        "nome":    "Belém",
        "pasta":   "data_belem",
        "arquivo": "belem_indicadores.csv",
    },
    # Para adicionar uma nova cidade, copie o bloco abaixo e preencha:
    # {
    #     "nome":    "Nome da Cidade",
    #     "pasta":   "data_nomecidade",
    #     "arquivo": "nomecidade_indicadores.csv",
    # },
]

# Anos futuros para projeção MLP
ANOS_PROJECAO = [2030, 2040]

# Parâmetros do modelo MLP (mesmos do notebook)
HIDDEN_LAYERS = (10, 10)
RANDOM_STATE  = 42
MAX_ITER      = 5000

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     FIM DO BLOCO DE CONFIGURAÇÃO                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ═══════════════════════════════════════════════════════════════════════════

def _raiz_repo() -> str:
    """Retorna o diretório raiz do repositório (onde app.py está)."""
    return os.path.dirname(os.path.abspath(__file__))


@st.cache_data(show_spinner=False)
def carregar_dados(pasta: str, arquivo: str) -> pd.DataFrame | None:
    """Lê o CSV de indicadores gerado pelo notebook. Retorna None se não existir."""
    caminho = os.path.join(_raiz_repo(), pasta, arquivo)
    if not os.path.exists(caminho):
        return None
    df = pd.read_csv(caminho)
    # Garante coluna numérica limpa
    if "valor" in df.columns:
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df


def _treinar_mlp(anos: list[int], valores: list[float],
                 anos_alvo: list[int]) -> list[float]:
    """Treina MLP igual ao notebook e retorna previsões para anos_alvo."""
    X = np.array(anos).reshape(-1, 1)
    y = np.array(valores).reshape(-1, 1)
    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X)
    ys = sy.fit_transform(y)
    m = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu", solver="lbfgs",
        max_iter=MAX_ITER, random_state=RANDOM_STATE,
    )
    m.fit(Xs, ys.ravel())
    Xa = sx.transform(np.array(anos_alvo).reshape(-1, 1))
    return sy.inverse_transform(m.predict(Xa).reshape(-1, 1)).ravel().tolist()


def _curva_mlp(anos: list[int], valores: list[float]) -> tuple[list[int], list[float]]:
    """Gera curva contínua para o gráfico (1990–2045)."""
    anos_curva = list(range(min(anos) - 2, 2046))
    vals_curva = _treinar_mlp(anos, valores, anos_curva)
    return anos_curva, vals_curva


# ═══════════════════════════════════════════════════════════════════════════
# GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════

def fig_serie(titulo: str, anos: list[int], valores: list[float],
              ylabel: str, municipio: str) -> go.Figure:
    """Gráfico de linha com dados reais + curva MLP + projeções."""
    anos_proj = ANOS_PROJECAO
    vals_proj = _treinar_mlp(anos, valores, anos_proj)
    anos_curva, vals_curva = _curva_mlp(anos, valores)

    fig = go.Figure()

    # Curva MLP
    fig.add_trace(go.Scatter(
        x=anos_curva, y=vals_curva,
        mode="lines", name="Curva MLP",
        line=dict(color="#7B1FA2", width=2, dash="solid"), opacity=0.8,
    ))
    # Dados reais
    fig.add_trace(go.Scatter(
        x=anos, y=valores,
        mode="lines+markers", name="Censos IBGE",
        line=dict(color="#1565C0", width=3),
        marker=dict(size=11, color="#0D47A1"),
    ))
    # Projeções
    fig.add_trace(go.Scatter(
        x=anos_proj, y=vals_proj,
        mode="markers+text", name="Projeção",
        text=[f"{v:,.1f}" for v in vals_proj],
        textposition="top center",
        marker=dict(size=14, color="#E65100", symbol="star"),
    ))

    fig.update_layout(
        template="plotly_white", height=420,
        title=dict(text=f"<b>{titulo}</b> — {municipio}", x=0.01),
        xaxis_title="Ano",
        yaxis_title=ylabel,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=60, b=40),
    )
    if max(valores) > 1000:
        fig.update_yaxes(tickformat=",")
    return fig


def fig_barras_todos(df: pd.DataFrame, municipio: str) -> go.Figure:
    """Gráfico de barras agrupadas: todos os indicadores × anos disponíveis."""
    fig = px.bar(
        df, x="indicador_nome", y="valor", color="ano",
        barmode="group", text_auto=True,
        title=f"Visão geral — {municipio}",
        color_continuous_scale="Blues",
        labels={"indicador_nome": "Indicador", "valor": "Valor", "ano": "Ano"},
    )
    fig.update_layout(
        template="plotly_white", height=500,
        xaxis_tickangle=-35,
        margin=dict(l=40, r=20, t=60, b=120),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SEÇÕES DO PAINEL
# ═══════════════════════════════════════════════════════════════════════════

def _css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1rem; max-width: 1200px; }
          .municipio-header {
            background: linear-gradient(135deg, #E3F2FD 0%, #fff 60%, #FFF8E1 100%);
            padding: 1.2rem 1.5rem;
            border-radius: 14px;
            border: 1px solid #BBDEFB;
            margin-bottom: 1rem;
          }
          .badge {
            display: inline-block;
            background: #1565C0;
            color: white !important;
            padding: 0.18rem 0.6rem;
            border-radius: 999px;
            font-size: 0.78rem;
          }
          .dado-ausente {
            background: #FFF3E0;
            border-left: 4px solid #FB8C00;
            padding: 0.8rem 1rem;
            border-radius: 6px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_municipio(cfg: dict) -> None:
    """Renderiza a aba completa de um município."""
    nome   = cfg["nome"]
    df     = carregar_dados(cfg["pasta"], cfg["arquivo"])

    st.markdown(
        f"""
        <div class="municipio-header">
          <span class="badge">Município</span>
          <h2 style="margin: 0.4rem 0 0.2rem; color:#0D47A1;">{nome}</h2>
          <p style="margin:0; color:#475569; font-size:0.95rem;">
            Dados do IBGE (Censos 1991–2022) com projeções MLP para
            {" e ".join(str(a) for a in ANOS_PROJECAO)}.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df is None:
        st.markdown(
            f"""
            <div class="dado-ausente">
              <strong>⚠️ Arquivo não encontrado:</strong>
              <code>{cfg['pasta']}/{cfg['arquivo']}</code><br>
              Execute o notebook para gerar os dados e faça push para o repositório.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Métricas de resumo ────────────────────────────────────────────────
    n_indicadores = df["indicador_id"].nunique()
    anos_disp     = sorted(df["ano"].dropna().unique().astype(int))
    grupos        = df["grupo_censo"].unique()

    m1, m2, m3 = st.columns(3)
    m1.metric("Indicadores", n_indicadores)
    m2.metric("Anos disponíveis", f"{anos_disp[0]}–{anos_disp[-1]}")
    m3.metric("Grupos", len(grupos))

    st.divider()

    # ── Visão geral (barras) ──────────────────────────────────────────────
    with st.expander("📊 Visão geral — todos os indicadores", expanded=False):
        st.plotly_chart(fig_barras_todos(df, nome), use_container_width=True)

    st.divider()

    # ── Tabela de dados brutos ────────────────────────────────────────────
    with st.expander("🗂️ Tabela de dados brutos", expanded=False):
        st.dataframe(
            df.sort_values(["indicador_nome", "ano"]),
            use_container_width=True,
            hide_index=True,
        )
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Baixar CSV",
            data=csv,
            file_name=cfg["arquivo"],
            mime="text/csv",
        )

    st.divider()

    # ── Gráficos por indicador ────────────────────────────────────────────
    st.subheader("Séries históricas e projeções MLP")
    st.caption(
        "Selecione um indicador no menu abaixo ou percorra todos pelo botão."
    )

    indicadores = sorted(df["indicador_nome"].unique())
    escolha = st.selectbox(
        "Indicador",
        options=["(Todos)"] + indicadores,
        key=f"sel_{nome}",
    )

    def _render_grafico_indicador(ind_nome: str) -> None:
        sub = df[df["indicador_nome"] == ind_nome].sort_values("ano")
        anos   = sub["ano"].astype(int).tolist()
        vals   = sub["valor"].tolist()
        ylabel = sub["unidade_medida"].iloc[0] if "unidade_medida" in sub.columns else "Valor"
        grupo  = sub["grupo_censo"].iloc[0] if "grupo_censo" in sub.columns else ""

        if len(anos) < 2 or any(np.isnan(v) for v in vals):
            st.warning(
                f"**{ind_nome}** — série com menos de 2 pontos válidos; projeção indisponível.",
                icon="⚠️",
            )
            return

        st.plotly_chart(
            fig_serie(ind_nome, anos, vals, ylabel, nome),
            use_container_width=True,
        )

        # Tabela de projeções
        vals_proj = _treinar_mlp(anos, vals, ANOS_PROJECAO)
        df_proj = pd.DataFrame({
            "Ano": ANOS_PROJECAO,
            f"Projeção MLP ({ylabel})": [round(v, 2) for v in vals_proj],
        })
        col_tab, col_esp = st.columns([1, 2])
        col_tab.caption(f"Grupo: **{grupo}** | Censos na série: {anos}")
        col_tab.dataframe(df_proj, hide_index=True, use_container_width=True)
        st.divider()

    if escolha == "(Todos)":
        for ind in indicadores:
            _render_grafico_indicador(ind)
    else:
        _render_grafico_indicador(escolha)


def render_comparativo(municipios_carregados: list[dict]) -> None:
    """Aba que compara o mesmo indicador entre municípios."""
    st.header("Comparativo entre municípios")
    st.caption(
        "Selecione um indicador para visualizar a evolução histórica "
        "e as projeções MLP lado a lado."
    )

    # Coleta todos os indicadores disponíveis em pelo menos um município
    todos_indicadores: set[str] = set()
    dfs: dict[str, pd.DataFrame] = {}
    for cfg in municipios_carregados:
        df = carregar_dados(cfg["pasta"], cfg["arquivo"])
        if df is not None:
            dfs[cfg["nome"]] = df
            todos_indicadores.update(df["indicador_nome"].unique())

    if not dfs:
        st.warning("Nenhum dado carregado ainda. Execute o notebook e faça push.")
        return

    ind_escolhido = st.selectbox(
        "Indicador para comparar",
        sorted(todos_indicadores),
        key="sel_comparativo",
    )

    fig = go.Figure()
    cores = px.colors.qualitative.Bold
    tem_dados = False

    for i, (mun, df) in enumerate(dfs.items()):
        sub = df[df["indicador_nome"] == ind_escolhido].sort_values("ano")
        if sub.empty:
            continue
        anos = sub["ano"].astype(int).tolist()
        vals = sub["valor"].tolist()
        if len(anos) < 2 or any(np.isnan(v) for v in vals):
            continue

        cor = cores[i % len(cores)]
        ylabel = sub["unidade_medida"].iloc[0] if "unidade_medida" in sub.columns else "Valor"

        # Dados reais
        fig.add_trace(go.Scatter(
            x=anos, y=vals,
            mode="lines+markers", name=f"{mun} — Censos",
            line=dict(color=cor, width=3),
            marker=dict(size=10),
        ))
        # Projeções
        vals_proj = _treinar_mlp(anos, vals, ANOS_PROJECAO)
        fig.add_trace(go.Scatter(
            x=ANOS_PROJECAO, y=vals_proj,
            mode="markers+text", name=f"{mun} — Projeção",
            text=[f"{v:,.1f}" for v in vals_proj],
            textposition="top center",
            marker=dict(size=13, symbol="star", color=cor),
        ))
        tem_dados = True

    if not tem_dados:
        st.info(f"Nenhum município tem dados para **{ind_escolhido}**.")
        return

    fig.update_layout(
        template="plotly_white", height=460,
        title=dict(text=f"<b>{ind_escolhido}</b> — comparativo", x=0.01),
        xaxis_title="Ano", yaxis_title=ylabel,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabela comparativa
    linhas = []
    for mun, df in dfs.items():
        sub = df[df["indicador_nome"] == ind_escolhido].sort_values("ano")
        if sub.empty:
            continue
        anos = sub["ano"].astype(int).tolist()
        vals = sub["valor"].tolist()
        if len(anos) < 2 or any(np.isnan(v) for v in vals):
            continue
        vals_proj = _treinar_mlp(anos, vals, ANOS_PROJECAO)
        for ano, val in zip(anos, vals):
            linhas.append({"Município": mun, "Ano": ano, "Valor (censo)": val, "Tipo": "Censo"})
        for ano, val in zip(ANOS_PROJECAO, vals_proj):
            linhas.append({"Município": mun, "Ano": ano, "Valor (projeção MLP)": round(val, 2), "Tipo": "Projeção"})

    st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title=TITULO_PAINEL,
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="📊",
    )
    _css()

    with st.sidebar:
        st.markdown(f"### {TITULO_PAINEL}")
        st.caption("TCC — Projeções via MLP (sklearn)")
        st.divider()
        st.markdown(
            "**Como adicionar uma cidade:**\n"
            "1. Rode o notebook coringa para o município.\n"
            "2. Faça push da pasta `data_<cidade>/` com o CSV.\n"
            "3. Adicione a entrada em `MUNICIPIOS` no topo de `app.py`.\n"
        )

    # Abas dinâmicas: uma por município + aba de comparativo
    nomes_abas = [cfg["nome"] for cfg in MUNICIPIOS] + ["🔀 Comparativo"]
    abas = st.tabs(nomes_abas)

    for aba, cfg in zip(abas[:-1], MUNICIPIOS):
        with aba:
            render_municipio(cfg)

    with abas[-1]:
        render_comparativo(MUNICIPIOS)


if __name__ == "__main__":
    main()
