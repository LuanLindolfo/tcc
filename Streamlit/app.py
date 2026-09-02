# -*- coding: utf-8 -*-
"""
app.py — Painel Streamlit TCC (iSaci)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Configuração da página e identidade visual
st.set_page_config(
    page_title="iSaci — Projeções Municipais IBGE",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌿",
)

TITULO_PAINEL = "Projeções Municipais IBGE"
LOGO_PATH = "logo_isaci.png" # Certifique-se de salvar a logo com este nome na raiz do projeto

MUNICIPIOS = [
    {"nome": "Castanhal", "pasta": "data", "arquivo": "indicadores_castanhal_tratados.csv"},
    {"nome": "Belém", "pasta": "data", "arquivo": "indicadores_belem_tratados.csv"},
    {"nome": "Ananindeua", "pasta": "data", "arquivo": "indicadores_ananindeua_tratados.csv"},
    {"nome": "Santarém", "pasta": "data", "arquivo": "indicadores_santarem_tratados.csv"},
    {"nome": "Parauapebas", "pasta": "data", "arquivo": "indicadores_parauapebas_tratados.csv"},
    {"nome": "Marabá", "pasta": "data", "arquivo": "indicadores_maraba_tratados.csv"},
]

NIVEIS_ACIMA_PARA_DADOS = 1
ANOS_PROJECAO = [2030, 2040]
ATIVACAO_FALLBACK = "relu"
SOLVER_FALLBACK   = "lbfgs"
HIDDEN_LAYERS     = (10, 10)
RANDOM_STATE      = 42
MAX_ITER          = 500


def _css() -> None:
    """Aplica o tema visual baseado na identidade iSaci."""
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.5rem; max-width: 1200px; }
          .isaci-card {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 6px solid #80C225;
            color: #FFFFFF;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
          }
          .isaci-card h2 { color: #80C225 !important; margin-top: 0; }
          .badge-isaci {
            background-color: #80C225;
            color: #0F172A !important;
            font-weight: bold;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
          }
          .badge-auto {
            background: #10B981;
            color: white !important;
            padding: 0.12rem 0.5rem;
            border-radius: 999px;
            font-size: 0.72rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _raiz_repo() -> str:
    caminho = os.path.dirname(os.path.abspath(__file__))
    for _ in range(NIVEIS_ACIMA_PARA_DADOS):
        caminho = os.path.dirname(caminho)
    return caminho


def _arquivo_projecoes(arquivo_historico: str) -> str:
    nome = arquivo_historico
    if nome.startswith("indicadores_") and nome.endswith("_tratados.csv"):
        meio = nome[len("indicadores_"):-len("_tratados.csv")]
        anos_tag = "_".join(str(a) for a in ANOS_PROJECAO)
        return f"projecoes_{meio}_{anos_tag}.csv"
    return ""


@st.cache_data(show_spinner=False, ttl="24h")
def carregar_dados(pasta: str, arquivo: str) -> pd.DataFrame | None:
    caminho = os.path.join(_raiz_repo(), pasta, arquivo)
    if not os.path.exists(caminho):
        return None
    df = pd.read_csv(caminho)
    if "valor" in df.columns:
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    if "ativacao" not in df.columns:
        df["ativacao"] = ATIVACAO_FALLBACK
    if "solver" not in df.columns:
        df["solver"] = SOLVER_FALLBACK
    if "auto_selecionado" not in df.columns:
        df["auto_selecionado"] = False
    if "loocv_mae" not in df.columns:
        df["loocv_mae"] = np.nan
    if "indicador_id" not in df.columns:
        df["indicador_id"] = df.get("indicador_nome", "")
    return df


@st.cache_data(show_spinner=False, ttl="24h")
def carregar_projecoes(pasta: str, arquivo_historico: str) -> pd.DataFrame | None:
    nome_proj = _arquivo_projecoes(arquivo_historico)
    if not nome_proj:
        return None
    caminho = os.path.join(_raiz_repo(), pasta, nome_proj)
    if not os.path.exists(caminho):
        return None
    df = pd.read_csv(caminho)
    if "valor_previsto" in df.columns:
        df["valor_previsto"] = pd.to_numeric(df["valor_previsto"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def _treinar_mlp_cached(anos: tuple[int, ...], valores: tuple[float, ...], 
                        anos_alvo: tuple[int, ...], ativacao: str, solver: str) -> list[float]:
    X = np.array(anos).reshape(-1, 1)
    y = np.array(valores).reshape(-1, 1)
    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X)
    ys = sy.fit_transform(y)
    m = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation=ativacao, solver=solver,
        max_iter=MAX_ITER, random_state=RANDOM_STATE,
    )
    m.fit(Xs, ys.ravel())
    Xa = sx.transform(np.array(anos_alvo).reshape(-1, 1))
    return sy.inverse_transform(m.predict(Xa).reshape(-1, 1)).ravel().tolist()


def _treinar_mlp(anos: list[int], valores: list[float], anos_alvo: list[int],
                 ativacao: str = ATIVACAO_FALLBACK, solver: str = SOLVER_FALLBACK) -> list[float]:
    return _treinar_mlp_cached(tuple(anos), tuple(valores), tuple(anos_alvo), ativacao, solver)


@st.cache_data(show_spinner=False)
def _curva_mlp_cached(anos: tuple[int, ...], valores: tuple[float, ...],
                       ativacao: str, solver: str) -> tuple[list[int], list[float]]:
    anos_curva = list(range(min(anos) - 2, 2046))
    vals_curva = _treinar_mlp_cached(anos, valores, tuple(anos_curva), ativacao, solver)
    return anos_curva, vals_curva


def _curva_mlp(anos: list[int], valores: list[float], ativacao: str, solver: str) -> tuple[list[int], list[float]]:
    return _curva_mlp_cached(tuple(anos), tuple(valores), ativacao, solver)


def _projecao_indicador(ind_id: str, anos: list[int], valores: list[float],
                        ativacao: str, solver: str,
                        df_proj_precalc: pd.DataFrame | None) -> list[float]:
    if df_proj_precalc is not None and "indicador_id" in df_proj_precalc.columns:
        sub = df_proj_precalc[df_proj_precalc["indicador_id"] == ind_id].sort_values("ano_previsto")
        if len(sub) == len(ANOS_PROJECAO) and list(sub["ano_previsto"]) == ANOS_PROJECAO:
            return sub["valor_previsto"].tolist()
    return _treinar_mlp(anos, valores, ANOS_PROJECAO, ativacao, solver)


def fig_serie(titulo: str, anos: list[int], valores: list[float],
              ylabel: str, municipio: str, ativacao: str, solver: str,
              vals_proj: list[float]) -> go.Figure:
    anos_curva, vals_curva = _curva_mlp(anos, valores, ativacao, solver)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=anos_curva, y=vals_curva,
        mode="lines", name=f"Curva MLP ({ativacao}/{solver})",
        line=dict(color="#80C225", width=2, dash="dash"), opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=anos, y=valores,
        mode="lines+markers", name="Censos IBGE",
        line=dict(color="#0F172A", width=3),
        marker=dict(size=10, color="#0F172A"),
    ))
    fig.add_trace(go.Scatter(
        x=ANOS_PROJECAO, y=vals_proj,
        mode="markers+text", name="Projeção",
        text=[f"{v:,.1f}" for v in vals_proj],
        textposition="top center",
        marker=dict(size=12, color="#E65100", symbol="star"),
    ))

    fig.update_layout(
        template="plotly_white", height=420,
        title=dict(text=f"<b>{titulo}</b> — {municipio}", x=0.01),
        xaxis_title="Ano", yaxis_title=ylabel,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=60, b=40),
    )
    if max(valores) > 1000:
        fig.update_yaxes(tickformat=",")
    return fig


def render_municipio(cfg: dict) -> None:
    nome = cfg["nome"]
    df = carregar_dados(cfg["pasta"], cfg["arquivo"])
    df_proj_precalc = carregar_projecoes(cfg["pasta"], cfg["arquivo"])

    st.markdown(
        f"""
        <div class="isaci-card">
          <span class="badge-isaci">MUNICÍPIO</span>
          <h2 style="margin: 0.5rem 0 0.2rem;">{nome}</h2>
          <p style="margin:0; color:#94A3B8; font-size:0.95rem;">
            Análise temporal IBGE (1991–2022) e projeções preditivas via Rede Neural (MLP).
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df is None:
        st.error(f"Arquivo de dados não localizado: {cfg['pasta']}/{cfg['arquivo']}")
        return

    indicadores = sorted(df["indicador_nome"].unique())
    escolha = st.selectbox("Selecione o Indicador", options=indicadores, key=f"sel_{nome}")

    sub = df[df["indicador_nome"] == escolha].sort_values("ano")
    anos = sub["ano"].astype(int).tolist()
    vals = sub["valor"].tolist()
    ylabel = sub["unidade_medida"].iloc[0] if "unidade_medida" in sub.columns else "Valor"
    ind_id = sub["indicador_id"].iloc[0]
    ativacao = sub["ativacao"].iloc[0]
    solver = sub["solver"].iloc[0]

    vals_proj = _projecao_indicador(ind_id, anos, vals, ativacao, solver, df_proj_precalc)

    st.plotly_chart(
        fig_serie(escolha, anos, vals, ylabel, nome, ativacao, solver, vals_proj),
        use_container_width=True,
    )


def main() -> None:
    _css()

    with st.sidebar:
        # Exibição da Logo do Instituto iSaci
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.title("iSaci")

        st.markdown("---")
        st.markdown(f"### {TITULO_PAINEL}")

        if st.button("🔄 Recarregar Dados", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.caption("Painel Preditivo de Dados Demográficos")

    nomes_abas = [cfg["nome"] for cfg in MUNICIPIOS]
    abas = st.tabs(nomes_abas)

    for aba, cfg in zip(abas, MUNICIPIOS):
        with aba:
            render_municipio(cfg)


if __name__ == "__main__":
    main()
