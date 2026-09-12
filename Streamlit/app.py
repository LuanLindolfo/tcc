# -*- coding: utf-8 -*-
"""
app.py — Painel Streamlit TCC (coringa multi-município)
Tema escuro refinado (identidade ISACI) + navegação lateral por município.
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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║              BLOCO DE CONFIGURAÇÃO — EDITE APENAS AQUI                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

TITULO_PAINEL = "Censo IBGE — Projeções Municipais"

MUNICIPIOS = [
    {
        "nome":    "Castanhal",
        "pasta":   "data",
        "arquivo": "indicadores_castanhal_tratados.csv",
    },
    {
        "nome":    "Belém",
        "pasta":   "data",
        "arquivo": "indicadores_belem_tratados.csv",
    },
    {
        "nome":    "Ananindeua",
        "pasta":   "data",
        "arquivo": "indicadores_ananindeua_tratados.csv",
    },
    {
        "nome":    "Santarém",
        "pasta":   "data",
        "arquivo": "indicadores_santarem_tratados.csv",
    },
    {
        "nome":    "Parauapebas",
        "pasta":   "data",
        "arquivo": "indicadores_parauapebas_tratados.csv",
    },
    {
        "nome":    "Abaetetuba",
        "pasta":   "data",
        "arquivo": "indicadores_abaetetuba_tratados.csv",
    },
    {
        "nome":    "Barcarena",
        "pasta":   "data",
        "arquivo": "indicadores_barcarena_tratados.csv",
    },
    {
        "nome":    "Cametá",
        "pasta":   "data",
        "arquivo": "indicadores_cametá_tratados.csv",
    },
    {
        "nome":    "Marabá",
        "pasta":   "data",
        "arquivo": "indicadores_maraba_tratados.csv",
    },
    {
        "nome":    "Altamira",
        "pasta":   "data",
        "arquivo": "indicadores_altamira_tratados.csv",
    },
    {
        "nome":    "Itaituba",
        "pasta":   "data",
        "arquivo": "indicadores_itaituba_tratados.csv",
    },
    {
        "nome":    "Bragança",
        "pasta":   "data",
        "arquivo": "indicadores_bragança_tratados.csv",
    },
]

NIVEIS_ACIMA_PARA_DADOS = 1
ANOS_PROJECAO = [2030, 2040]

ATIVACAO_FALLBACK = "relu"
SOLVER_FALLBACK   = "lbfgs"
HIDDEN_LAYERS     = (10, 10)
RANDOM_STATE      = 42

# OTIMIZAÇÃO: Reduzido de 5000 para 500. É suficiente para 4-5 pontos censitários.
MAX_ITER          = 500

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     FIM DO BLOCO DE CONFIGURAÇÃO                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════════
# LOGO — embutido em base64 para nunca depender de um arquivo externo no repo
# ═══════════════════════════════════════════════════════════════════════════

LOGO_MIME = "webp"
LOGO_BASE64 = "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"


# ═══════════════════════════════════════════════════════════════════════════
# IDENTIDADE VISUAL — tema escuro
# ═══════════════════════════════════════════════════════════════════════════

COR_FUNDO       = "#0B0F19"   # base quase-preta
COR_FUNDO_2     = "#0F1626"
COR_CARD        = "rgba(255,255,255,0.035)"
COR_CARD_BORDA  = "rgba(255,255,255,0.09)"
COR_TEXTO       = "#E4E9F2"
COR_TEXTO_SUAVE = "#94A3B8"

COR_AZUL        = "#3B82F6"   # acento primário (gradiente principal)
COR_ROXO        = "#8B5CF6"   # acento secundário (gradiente principal)
COR_MARCA       = "#8BC53F"   # verde do logo ISACI — acento institucional
COR_ACENTO      = "#F59E0B"   # âmbar — projeções / destaques
COR_SUCESSO     = "#34D399"
COR_ALERTA      = "#FBBF24"

PALETA_COMPARATIVO = [
    "#3B82F6", "#F59E0B", "#8B5CF6", "#F472B6",
    "#34D399", "#8BC53F", "#60A5FA", "#FB923C",
    "#38BDF8", "#A78BFA", "#4ADE80", "#F87171",
]


# ═══════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ═══════════════════════════════════════════════════════════════════════════

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


# OTIMIZAÇÃO: Função interna com @st.cache_data usando tuplas (hasháveis)
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
                 ativacao: str = ATIVACAO_FALLBACK,
                 solver: str = SOLVER_FALLBACK) -> list[float]:
    """Converte listas em tuplas para usar o cache do Streamlit."""
    return _treinar_mlp_cached(tuple(anos), tuple(valores), tuple(anos_alvo), ativacao, solver)


@st.cache_data(show_spinner=False)
def _curva_mlp_cached(anos: tuple[int, ...], valores: tuple[float, ...],
                       ativacao: str, solver: str) -> tuple[list[int], list[float]]:
    anos_curva = list(range(min(anos) - 2, 2046))
    vals_curva = _treinar_mlp_cached(anos, valores, tuple(anos_curva), ativacao, solver)
    return anos_curva, vals_curva


def _curva_mlp(anos: list[int], valores: list[float],
               ativacao: str, solver: str) -> tuple[list[int], list[float]]:
    return _curva_mlp_cached(tuple(anos), tuple(valores), ativacao, solver)


def _projecao_indicador(ind_id: str, anos: list[int], valores: list[float],
                        ativacao: str, solver: str,
                        df_proj_precalc: pd.DataFrame | None) -> list[float]:
    if df_proj_precalc is not None and "indicador_id" in df_proj_precalc.columns:
        sub = df_proj_precalc[df_proj_precalc["indicador_id"] == ind_id].sort_values("ano_previsto")
        if len(sub) == len(ANOS_PROJECAO) and list(sub["ano_previsto"]) == ANOS_PROJECAO:
            return sub["valor_previsto"].tolist()
    return _treinar_mlp(anos, valores, ANOS_PROJECAO, ativacao, solver)


# ═══════════════════════════════════════════════════════════════════════════
# GRÁFICOS (tema escuro)
# ═══════════════════════════════════════════════════════════════════════════

_FONTE = dict(family="Inter, -apple-system, sans-serif", color=COR_TEXTO)
_GRID  = "rgba(255,255,255,0.08)"


def _layout_base(fig: go.Figure, altura: int) -> None:
    fig.update_layout(
        template="plotly_dark",
        height=altura,
        font=_FONTE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(size=12, color=COR_TEXTO_SUAVE)),
    )
    fig.update_xaxes(gridcolor=_GRID, zeroline=False, linecolor=_GRID)
    fig.update_yaxes(gridcolor=_GRID, zeroline=False, linecolor=_GRID)


def fig_serie(titulo: str, anos: list[int], valores: list[float],
              ylabel: str, municipio: str, ativacao: str, solver: str,
              vals_proj: list[float]) -> go.Figure:
    anos_curva, vals_curva = _curva_mlp(anos, valores, ativacao, solver)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=anos_curva, y=vals_curva,
        mode="lines", name=f"Curva MLP ({ativacao}/{solver})",
        line=dict(color=COR_ROXO, width=2, dash="solid"), opacity=0.55,
    ))
    fig.add_trace(go.Scatter(
        x=anos, y=valores,
        mode="lines+markers", name="Censos IBGE",
        line=dict(color=COR_AZUL, width=3),
        marker=dict(size=10, color=COR_AZUL, line=dict(width=1.5, color=COR_FUNDO)),
    ))
    fig.add_trace(go.Scatter(
        x=ANOS_PROJECAO, y=vals_proj,
        mode="markers+text", name="Projeção",
        text=[f"{v:,.1f}" for v in vals_proj],
        textposition="top center",
        textfont=dict(color=COR_ACENTO),
        marker=dict(size=15, color=COR_ACENTO, symbol="star",
                    line=dict(width=1.2, color="#7C2D12")),
    ))

    _layout_base(fig, 430)
    fig.update_layout(
        title=dict(text=f"<b>{titulo}</b> — {municipio}", x=0.01,
                   font=dict(size=16, color=COR_TEXTO)),
        xaxis_title="Ano", yaxis_title=ylabel,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0,
                    font=dict(size=12, color=COR_TEXTO_SUAVE)),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=60, b=40),
    )
    if max(valores) > 1000:
        fig.update_yaxes(tickformat=",")
    return fig


def fig_barras_todos(df: pd.DataFrame, municipio: str) -> go.Figure:
    """Visão geral separada por grupo censitário — evita empilhar indicadores
    de escalas muito diferentes (ex.: percentuais x contagens) no mesmo eixo,
    o que tornava o gráfico anterior ilegível."""
    tem_grupo = "grupo_censo" in df.columns and df["grupo_censo"].notna().any()
    grupos = sorted(df["grupo_censo"].dropna().unique()) if tem_grupo else ["Indicadores"]
    n_grupos = max(len(grupos), 1)

    fig = px.bar(
        df, x="indicador_nome", y="valor", color="ano",
        barmode="group",
        facet_col="grupo_censo" if tem_grupo else None,
        facet_col_wrap=1,
        title=f"Visão geral — {municipio}",
        color_discrete_sequence=[COR_AZUL, COR_ROXO, COR_ACENTO, COR_MARCA, "#38BDF8"],
        labels={"indicador_nome": "Indicador", "valor": "Valor", "ano": "Ano"},
    )
    fig.for_each_annotation(lambda a: a.update(
        text=a.text.split("=")[-1], font=dict(size=13, color=COR_TEXTO)
    ))
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_traces(textposition="none",
                       hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:,.2f}<extra></extra>")

    _layout_base(fig, max(360 * n_grupos, 420))
    fig.update_layout(
        title=dict(font=dict(size=16, color=COR_TEXTO)),
        xaxis_tickangle=-30,
        margin=dict(l=40, r=20, t=60, b=100),
    )
    return fig


def fig_comparativo(dfs: dict[str, pd.DataFrame], dfs_proj: dict, ind_escolhido: str) -> tuple[go.Figure, str, bool]:
    """Gráfico comparativo com legenda enxuta (1 entrada por município,
    a projeção herda a cor e fica agrupada, sem poluir a legenda)."""
    fig = go.Figure()
    cores = PALETA_COMPARATIVO
    tem_dados = False
    ylabel = "Valor"

    for i, (mun, df) in enumerate(dfs.items()):
        sub = df[df["indicador_nome"] == ind_escolhido].sort_values("ano")
        if sub.empty:
            continue
        anos = sub["ano"].astype(int).tolist()
        vals = sub["valor"].tolist()
        if len(anos) < 2 or any(np.isnan(v) for v in vals):
            continue

        ind_id   = sub["indicador_id"].iloc[0]
        ativacao = sub["ativacao"].iloc[0]
        solver   = sub["solver"].iloc[0]
        ylabel   = sub["unidade_medida"].iloc[0] if "unidade_medida" in sub.columns else "Valor"
        cor      = cores[i % len(cores)]

        fig.add_trace(go.Scatter(
            x=anos, y=vals,
            mode="lines+markers", name=mun,
            legendgroup=mun, showlegend=True,
            line=dict(color=cor, width=3),
            marker=dict(size=8, line=dict(width=1.3, color=COR_FUNDO)),
            hovertemplate=f"<b>{mun}</b> — %{{x}}: %{{y:,.2f}}<extra></extra>",
        ))

        vals_proj = _projecao_indicador(
            ind_id, anos, vals, ativacao, solver, dfs_proj.get(mun)
        )
        fig.add_trace(go.Scatter(
            x=ANOS_PROJECAO, y=vals_proj,
            mode="markers", name=mun,
            legendgroup=mun, showlegend=False,
            marker=dict(size=12, symbol="star", color=cor,
                        line=dict(width=1, color=COR_TEXTO)),
            hovertemplate=(f"<b>{mun}</b> — projeção %{{x}}: %{{y:,.2f}} "
                           f"({ativacao}/{solver})<extra></extra>"),
        ))
        tem_dados = True

    _layout_base(fig, 500)
    fig.update_layout(
        title=dict(text=f"<b>{ind_escolhido}</b>", x=0.01,
                   font=dict(size=16, color=COR_TEXTO)),
        xaxis_title="Ano", yaxis_title=ylabel,
        legend=dict(
            orientation="v",
            yanchor="top", y=1,
            xanchor="left", x=1.015,
            font=dict(size=11.5, color=COR_TEXTO_SUAVE),
            bgcolor="rgba(255,255,255,0.03)",
            bordercolor=COR_CARD_BORDA, borderwidth=1,
            tracegroupgap=2,
        ),
        hovermode="closest",
        margin=dict(l=50, r=170, t=60, b=40),
    )
    return fig, ylabel, tem_dados


# ═══════════════════════════════════════════════════════════════════════════
# ESTILO GLOBAL
# ═══════════════════════════════════════════════════════════════════════════

def _css() -> None:
    st.markdown(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

          html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif !important; }}

          .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
            background:
              radial-gradient(circle at 15% 0%, rgba(59,130,246,0.10), transparent 45%),
              radial-gradient(circle at 90% 15%, rgba(139,92,246,0.08), transparent 40%),
              {COR_FUNDO} !important;
          }}
          [data-testid="stHeader"] {{ background: transparent !important; }}
          .block-container {{ padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1240px; }}

          h1, h2, h3, h4, p, span, label, li,
          [data-testid="stMarkdownContainer"] {{ color: {COR_TEXTO}; }}

          /* ---------- Cabeçalho de página ---------- */
          .painel-header {{
            position: relative;
            background: {COR_CARD};
            border: 1px solid {COR_CARD_BORDA};
            border-radius: 20px;
            padding: 1.7rem 2rem;
            margin-bottom: 1.4rem;
            backdrop-filter: blur(6px);
            overflow: hidden;
          }}
          .painel-header::before {{
            content: "";
            position: absolute; inset: 0;
            background: linear-gradient(120deg, rgba(59,130,246,0.14), rgba(139,92,246,0.10) 50%, transparent 80%);
            pointer-events: none;
          }}
          .painel-header-top {{
            display: flex; align-items: center; justify-content: space-between;
            position: relative; z-index: 1;
          }}
          .painel-header-logo {{ height: 26px; opacity: 0.95; }}
          .painel-header h2 {{
            margin: 0.5rem 0 0.25rem;
            background: linear-gradient(90deg, #FFFFFF 0%, #C7D6FE 100%);
            -webkit-background-clip: text; background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800; letter-spacing: -0.01em; font-size: 1.7rem;
            position: relative; z-index: 1;
          }}
          .painel-header p {{
            margin: 0; color: {COR_TEXTO_SUAVE} !important;
            font-size: 0.93rem; max-width: 760px; line-height: 1.55;
            position: relative; z-index: 1;
          }}

          .badge {{
            display: inline-block;
            background: rgba(59,130,246,0.14);
            color: #93C5FD !important;
            padding: 0.22rem 0.75rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            border: 1px solid rgba(59,130,246,0.35);
          }}
          .badge-auto {{
            display: inline-block;
            background: rgba(52,211,153,0.14);
            color: {COR_SUCESSO} !important;
            padding: 0.16rem 0.6rem;
            border-radius: 999px;
            font-size: 0.72rem; font-weight: 600;
            margin-left: 0.45rem;
            border: 1px solid rgba(52,211,153,0.3);
          }}
          .badge-fixo {{
            display: inline-block;
            background: rgba(148,163,184,0.14);
            color: {COR_TEXTO_SUAVE} !important;
            padding: 0.16rem 0.6rem;
            border-radius: 999px;
            font-size: 0.72rem; font-weight: 600;
            margin-left: 0.45rem;
            border: 1px solid rgba(148,163,184,0.25);
          }}

          .dado-ausente {{
            background: rgba(251,191,36,0.08);
            border-left: 4px solid {COR_ALERTA};
            padding: 0.9rem 1.1rem;
            border-radius: 10px;
            color: {COR_TEXTO} !important;
          }}
          .dado-ausente code {{ color: {COR_ALERTA}; }}

          /* ---------- Métricas ---------- */
          div[data-testid="stMetric"] {{
            background: {COR_CARD};
            border: 1px solid {COR_CARD_BORDA};
            border-radius: 14px;
            padding: 0.9rem 1.1rem 0.7rem;
          }}
          div[data-testid="stMetricLabel"] {{
            color: {COR_TEXTO_SUAVE} !important; font-weight: 500;
            text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.04em;
          }}
          div[data-testid="stMetricValue"] {{ color: {COR_TEXTO} !important; font-weight: 700; }}

          /* ---------- Sidebar ---------- */
          section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0D1220 0%, #0A0E17 100%) !important;
            border-right: 1px solid {COR_CARD_BORDA};
          }}
          section[data-testid="stSidebar"] * {{ color: {COR_TEXTO}; }}
          .sidebar-brand {{
            display: flex; align-items: center; gap: 0.7rem;
            background: {COR_CARD}; border: 1px solid {COR_CARD_BORDA};
            border-radius: 14px; padding: 0.85rem 1rem; margin-bottom: 1.1rem;
          }}
          .sidebar-brand img {{ height: 30px; }}
          .sidebar-brand-title {{
            font-size: 0.82rem; font-weight: 700; color: {COR_TEXTO}; line-height: 1.25;
          }}
          .sidebar-eyebrow {{
            font-size: 0.7rem; font-weight: 700; letter-spacing: 0.09em;
            text-transform: uppercase; color: {COR_TEXTO_SUAVE};
            margin: 0.2rem 0 0.5rem;
          }}

          /* ---------- Inputs ---------- */
          [data-baseweb="input"], [data-baseweb="select"] > div {{
            background-color: rgba(255,255,255,0.04) !important;
            border-radius: 10px !important;
            border-color: {COR_CARD_BORDA} !important;
          }}
          div[role="radiogroup"] label {{
            border-radius: 10px;
            padding: 0.35rem 0.6rem;
            margin-bottom: 0.15rem;
            transition: background-color 0.15s ease;
          }}
          div[role="radiogroup"] label:hover {{ background-color: rgba(59,130,246,0.10); }}
          div[role="radiogroup"] label[data-baseweb="radio"] div:first-child {{
            border-color: {COR_AZUL} !important;
          }}

          .stButton > button {{
            border-radius: 10px !important;
            border: 1px solid {COR_CARD_BORDA} !important;
            background: rgba(255,255,255,0.03) !important;
            color: {COR_TEXTO} !important;
            font-weight: 600 !important;
          }}
          .stButton > button:hover {{
            border-color: {COR_AZUL} !important;
            color: #93C5FD !important;
          }}

          /* ---------- Expanders / cartões ---------- */
          div[data-testid="stExpander"] {{
            border: 1px solid {COR_CARD_BORDA} !important;
            border-radius: 14px !important;
            background: {COR_CARD} !important;
          }}
          div[data-testid="stExpander"] summary {{ font-weight: 600; color: {COR_TEXTO}; }}

          hr {{ margin: 1.1rem 0; border-color: {COR_CARD_BORDA}; opacity: 0.6; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(badge: str, titulo: str, descricao: str) -> None:
    logo_html = (
        f'<img class="painel-header-logo" src="data:image/{LOGO_MIME};base64,{LOGO_BASE64}">'
        if LOGO_BASE64 and not LOGO_BASE64.startswith("__") else ""
    )
    st.markdown(
        f"""
        <div class="painel-header">
          <div class="painel-header-top">
            <span class="badge">{badge}</span>
            {logo_html}
          </div>
          <h2>{titulo}</h2>
          <p>{descricao}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SEÇÕES DO PAINEL
# ═══════════════════════════════════════════════════════════════════════════

def render_municipio(cfg: dict) -> None:
    nome = cfg["nome"]
    df   = carregar_dados(cfg["pasta"], cfg["arquivo"])
    df_proj_precalc = carregar_projecoes(cfg["pasta"], cfg["arquivo"])

    _render_header(
        "Município",
        nome,
        "Dados do IBGE (Censos 1991–2022) com projeções MLP para "
        f"{' e '.join(str(a) for a in ANOS_PROJECAO)}, usando a "
        "ativação/solver escolhida automaticamente por indicador "
        "(validação leave-one-out) no notebook.",
    )

    if df is None:
        caminho_esperado = os.path.join(_raiz_repo(), cfg["pasta"], cfg["arquivo"])
        pasta_absoluta   = os.path.join(_raiz_repo(), cfg["pasta"])

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

        with st.expander("🔎 Diagnóstico — o que existe de fato no repositório", expanded=True):
            st.code(f"Caminho esperado:\n{caminho_esperado}", language="text")

            if not os.path.isdir(pasta_absoluta):
                st.error(f"A pasta '{cfg['pasta']}/' nem existe no repositório (ao lado de app.py).")
                st.write("Pastas encontradas na raiz do repositório:")
                st.code("\n".join(sorted(os.listdir(_raiz_repo()))) or "(vazio)", language="text")
            else:
                arquivos_na_pasta = sorted(os.listdir(pasta_absoluta))
                st.write(f"Arquivos encontrados em `{cfg['pasta']}/`:")
                st.code("\n".join(arquivos_na_pasta) or "(pasta vazia)", language="text")

                alvo = cfg["arquivo"].lower()
                parecidos = [
                    a for a in arquivos_na_pasta
                    if a.lower().replace('í', 'i').replace('é', 'e').replace('á', 'a')
                       == alvo.replace('í', 'i').replace('é', 'e').replace('á', 'a')
                ]
                if parecidos:
                    st.warning(
                        f"Existe um arquivo parecido, mas com nome diferente: "
                        f"`{parecidos[0]}` — confira acentos, maiúsculas ou "
                        f"underline vs. espaço, e ajuste `MUNICIPIOS` em app.py "
                        f"ou renomeie o CSV no repositório para bater exatamente."
                    )
        return

    n_indicadores = df["indicador_id"].nunique()
    anos_disp     = sorted(df["ano"].dropna().unique().astype(int))
    grupos        = df["grupo_censo"].unique() if "grupo_censo" in df.columns else []
    n_auto        = df.drop_duplicates("indicador_id")["auto_selecionado"].sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Indicadores", n_indicadores)
    m2.metric("Anos disponíveis", f"{anos_disp[0]}–{anos_disp[-1]}" if anos_disp else "—")
    m3.metric("Grupos", len(grupos))
    m4.metric("Modelo auto-selecionado", f"{int(n_auto)}/{n_indicadores}")

    if df_proj_precalc is None:
        st.caption(
            "ℹ️ CSV de projeções do notebook ainda não encontrado nesta pasta — "
            "as previsões abaixo estão sendo recalculadas em tempo real pelo app "
            "(mesma ativação/solver salva por indicador)."
        )

    st.divider()

    with st.expander("📊 Visão geral — todos os indicadores", expanded=False):
        st.caption("Separado por grupo censitário para manter escalas comparáveis.")
        st.plotly_chart(fig_barras_todos(df, nome), use_container_width=True)

    st.divider()

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

    st.subheader("Séries históricas e projeções MLP")
    st.caption("Selecione um indicador no menu abaixo.")

    indicadores = sorted(df["indicador_nome"].unique())

    # OTIMIZAÇÃO: Define o primeiro indicador como seleção padrão em vez de "(Todos)"
    # para evitar a geração simultânea massiva de gráficos Plotly.
    escolha = st.selectbox(
        "Indicador",
        options=indicadores + ["(Exibir Todos)"],
        key=f"sel_{nome}",
    )

    def _render_grafico_indicador(ind_nome: str) -> None:
        sub = df[df["indicador_nome"] == ind_nome].sort_values("ano")
        anos     = sub["ano"].astype(int).tolist()
        vals     = sub["valor"].tolist()
        ylabel   = sub["unidade_medida"].iloc[0] if "unidade_medida" in sub.columns else "Valor"
        grupo    = sub["grupo_censo"].iloc[0] if "grupo_censo" in sub.columns else ""
        ind_id   = sub["indicador_id"].iloc[0]
        ativacao = sub["ativacao"].iloc[0]
        solver   = sub["solver"].iloc[0]
        auto     = bool(sub["auto_selecionado"].iloc[0])
        loocv    = sub["loocv_mae"].iloc[0]

        if len(anos) < 2 or any(np.isnan(v) for v in vals):
            st.warning(
                f"**{ind_nome}** — série com menos de 2 pontos válidos; projeção indisponível.",
                icon="⚠️",
            )
            return

        badge_modelo = (
            f'<span class="badge-auto">auto — LOOCV MAE={loocv:,.2f}</span>'
            if auto and pd.notna(loocv)
            else f'<span class="badge-fixo">modelo fixo</span>'
        )
        st.markdown(
            f"**{ativacao} / {solver}** {badge_modelo}",
            unsafe_allow_html=True,
        )

        vals_proj = _projecao_indicador(ind_id, anos, vals, ativacao, solver, df_proj_precalc)

        st.plotly_chart(
            fig_serie(ind_nome, anos, vals, ylabel, nome, ativacao, solver, vals_proj),
            use_container_width=True,
        )

        df_proj_tabela = pd.DataFrame({
            "Ano": ANOS_PROJECAO,
            f"Projeção MLP ({ylabel})": [round(v, 2) for v in vals_proj],
        })
        col_tab, col_esp = st.columns([1, 2])
        col_tab.caption(f"Grupo: **{grupo}** | Censos na série: {anos}")
        col_tab.dataframe(df_proj_tabela, hide_index=True, use_container_width=True)
        st.divider()

    if escolha == "(Exibir Todos)":
        for ind in indicadores:
            _render_grafico_indicador(ind)
    else:
        _render_grafico_indicador(escolha)


def render_comparativo(municipios_carregados: list[dict]) -> None:
    _render_header(
        "Análise cruzada",
        "Comparativo entre municípios",
        "Selecione um indicador para visualizar a evolução histórica e as "
        "projeções lado a lado. Cada município usa a ativação/solver "
        "escolhida individualmente pelo seu próprio notebook. ★ marca as "
        f"projeções para {' e '.join(str(a) for a in ANOS_PROJECAO)}.",
    )

    todos_indicadores: set[str] = set()
    dfs: dict[str, pd.DataFrame] = {}
    dfs_proj: dict[str, pd.DataFrame | None] = {}
    for cfg in municipios_carregados:
        df = carregar_dados(cfg["pasta"], cfg["arquivo"])
        if df is not None:
            dfs[cfg["nome"]] = df
            dfs_proj[cfg["nome"]] = carregar_projecoes(cfg["pasta"], cfg["arquivo"])
            todos_indicadores.update(df["indicador_nome"].unique())

    if not dfs:
        st.warning("Nenhum dado carregado ainda. Execute o notebook e faça push.")
        return

    ind_escolhido = st.selectbox(
        "Indicador para comparar",
        sorted(todos_indicadores),
        key="sel_comparativo",
    )

    fig, ylabel, tem_dados = fig_comparativo(dfs, dfs_proj, ind_escolhido)

    if not tem_dados:
        st.info(f"Nenhum município tem dados para **{ind_escolhido}**.")
        return

    st.plotly_chart(fig, use_container_width=True)

    linhas = []
    for mun, df in dfs.items():
        sub = df[df["indicador_nome"] == ind_escolhido].sort_values("ano")
        if sub.empty:
            continue
        anos = sub["ano"].astype(int).tolist()
        vals = sub["valor"].tolist()
        if len(anos) < 2 or any(np.isnan(v) for v in vals):
            continue
        ind_id   = sub["indicador_id"].iloc[0]
        ativacao = sub["ativacao"].iloc[0]
        solver   = sub["solver"].iloc[0]
        vals_proj = _projecao_indicador(
            ind_id, anos, vals, ativacao, solver, dfs_proj.get(mun)
        )
        for ano, val in zip(anos, vals):
            linhas.append({
                "Município": mun, "Ano": ano, "Valor": val,
                "Tipo": "Censo", "Modelo": "—",
            })
        for ano, val in zip(ANOS_PROJECAO, vals_proj):
            linhas.append({
                "Município": mun, "Ano": ano, "Valor": round(val, 2),
                "Tipo": "Projeção", "Modelo": f"{ativacao}/{solver}",
            })

    with st.expander("🗂️ Tabela consolidada (censo + projeções)", expanded=False):
        st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# NAVEGAÇÃO LATERAL
# ═══════════════════════════════════════════════════════════════════════════

def _sidebar_navegacao(nomes_municipios: list[str]) -> str:
    """Sidebar com marca, busca e seleção de município, substituindo as abas no topo."""
    logo_html = (
        f'<img src="data:image/{LOGO_MIME};base64,{LOGO_BASE64}">'
        if LOGO_BASE64 and not LOGO_BASE64.startswith("__") else ""
    )
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
          {logo_html}
          <div class="sidebar-brand-title">{TITULO_PAINEL}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("🔄 Recarregar dados", help="Limpa o cache após subir novos dados no GitHub",
                          use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown('<div class="sidebar-eyebrow">Navegar por</div>', unsafe_allow_html=True)

    busca = st.sidebar.text_input(
        "Buscar município",
        placeholder="🔎 Digite para filtrar…",
        label_visibility="collapsed",
    )

    opcoes = [n for n in nomes_municipios if busca.strip().lower() in n.lower()] if busca else list(nomes_municipios)

    if "pagina_atual" not in st.session_state:
        st.session_state["pagina_atual"] = nomes_municipios[0]

    pagina_atual = st.session_state["pagina_atual"]
    lista_radio = opcoes + ["🔀 Comparativo entre municípios"]
    default_idx = 0
    if pagina_atual in lista_radio:
        default_idx = lista_radio.index(pagina_atual)
    elif pagina_atual == "Comparativo":
        default_idx = len(lista_radio) - 1

    if not opcoes and busca:
        st.sidebar.caption("Nenhum município encontrado.")

    escolha = st.sidebar.radio(
        "Município",
        options=lista_radio,
        index=default_idx if lista_radio else 0,
        label_visibility="collapsed",
    )

    pagina = "Comparativo" if escolha == "🔀 Comparativo entre municípios" else escolha
    st.session_state["pagina_atual"] = pagina

    st.sidebar.divider()
    st.sidebar.caption(
        "TCC — Projeções via MLP (sklearn), modelo por indicador escolhido via LOOCV."
    )
    st.sidebar.info(
        "Os dados têm como base o Censo do IBGE, o SIDRA e o Panorama do "
        "Censo. Parte das informações é estimada, podendo haver margem de "
        "erro para mais ou para menos, dado o alto volume de dados "
        "pesquisados e o tempo de atualização das bases.",
        icon="ℹ️",
    )
    with st.sidebar.expander("➕ Como adicionar uma cidade"):
        st.markdown(
            "1. Rode o notebook coringa para o município.\n"
            "2. Faça push da pasta `data_<cidade>/` com os CSVs "
            "(`indicadores_..._tratados.csv` e, se já gerado, "
            "`projecoes_..._2030_2040.csv`).\n"
            "3. Adicione a entrada em `MUNICIPIOS` no topo de `app.py`.\n"
        )

    return pagina


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

    nomes_municipios = [cfg["nome"] for cfg in MUNICIPIOS]
    pagina = _sidebar_navegacao(nomes_municipios)

    if pagina == "Comparativo":
        render_comparativo(MUNICIPIOS)
    else:
        cfg = next((c for c in MUNICIPIOS if c["nome"] == pagina), MUNICIPIOS[0])
        render_municipio(cfg)


if __name__ == "__main__":
    main()
