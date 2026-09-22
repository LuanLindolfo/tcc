# -*- coding: utf-8 -*-
"""
app.py — Painel Streamlit TCC (coringa multi-município)
Versão com identidade visual estilo "Scholaris" (Dark Mode, Neon Accents).
"""

from __future__ import annotations

import base64
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

# Caminho do logo (relativo à raiz do repositório). 
# IMPORTANTE: Verifique se o arquivo e a pasta existem com este nome exato!
LOGO_ARQUIVO = "assets/logo_isaci_neon.webp"

MUNICIPIOS = [
    {
        "nome":    "Abaetetuba",
        "pasta":   "data",
        "arquivo": "indicadores_abaetetuba_tratados.csv",
    },
    {
        "nome":    "Acará",
        "pasta":   "data",
        "arquivo": "indicadores_acará_tratados.csv",
    },
    {
        "nome":    "Alenquer",
        "pasta":   "data",
        "arquivo": "indicadores_alenquer_tratados.csv",
    },
    {
        "nome":    "Altamira",
        "pasta":   "data",
        "arquivo": "indicadores_altamira_tratados.csv",
    },
    {
        "nome":    "Ananindeua",
        "pasta":   "data",
        "arquivo": "indicadores_ananindeua_tratados.csv",
    },
    {
        "nome":    "Augusto Corrêa",
        "pasta":   "data",
        "arquivo": "indicadores_augusto_corrêa_tratados.csv",
    },
    {
        "nome":    "Baião",
        "pasta":   "data",
        "arquivo": "indicadores_baião_tratados.csv",
    },
    {
        "nome":    "Barcarena",
        "pasta":   "data",
        "arquivo": "indicadores_barcarena_tratados.csv",
    },
    {
        "nome":    "Belém",
        "pasta":   "data",
        "arquivo": "indicadores_belem_tratados.csv",
    },
    {
        "nome":    "Benevides",
        "pasta":   "data",
        "arquivo": "indicadores_benevides_tratados.csv",
    },
    {
        "nome":    "Bragança",
        "pasta":   "data",
        "arquivo": "indicadores_braganca_tratados.csv",
    },
    {
        "nome":    "Breu Branco",
        "pasta":   "data",
        "arquivo": "indicadores_breu_branco_tratados.csv",
    },
    {
        "nome":    "Breves",
        "pasta":   "data",
        "arquivo": "indicadores_breves_tratados.csv",
    },
    {
        "nome":    "Cametá",
        "pasta":   "data",
        "arquivo": "indicadores_cametá_tratados.csv",
    },
    {
        "nome":    "Canãa dos Carajás",
        "pasta":   "data",
        "arquivo": "indicadores_canãa_dos_carajá_tratados.csv",
    },
    {
        "nome":    "Capanema",
        "pasta":   "data",
        "arquivo": "indicadores_capanema_tratados.csv",
    },
    {
        "nome":    "Capitão Poço",
        "pasta":   "data",
        "arquivo": "indicadores_capitão_poço_tratados.csv",
    },
    {
        "nome":    "Castanhal",
        "pasta":   "data",
        "arquivo": "indicadores_castanhal_tratados.csv",
    },
    {
        "nome":    "Conceição do Araguaia",
        "pasta":   "data",
        "arquivo": "indicadores_conceição_do_araguaia_tratados.csv",
    },
    {
        "nome":    "Curuçá",
        "pasta":   "data",
        "arquivo": "indicadores_curuçá_tratados.csv",
    },
    {
        "nome":    "Dom Eliseu",
        "pasta":   "data",
        "arquivo": "indicadores_dom_eliseu_tratados.csv",
    },
    {
        "nome":    "Igarapé-Miri",
        "pasta":   "data",
        "arquivo": "indicadores_igarapé_miri_tratados.csv",
    },
    {
        "nome":    "Itaituba",
        "pasta":   "data",
        "arquivo": "indicadores_itaituba_tratados.csv",
    },
    {
        "nome":    "Itupiranga",
        "pasta":   "data",
        "arquivo": "indicadores_itupiranga_tratados.csv",
    },
    {
        "nome":    "Juruti",
        "pasta":   "data",
        "arquivo": "indicadores_juruti_tratados.csv",
    },
    {
        "nome":    "Marabá",
        "pasta":   "data",
        "arquivo": "indicadores_maraba_tratados.csv",
    },
    {
        "nome":    "Marituba",
        "pasta":   "data",
        "arquivo": "indicadores_marituba_tratados.csv",
    },
    {
        "nome":    "Moju",
        "pasta":   "data",
        "arquivo": "indicadores_moju_tratados.csv",
    },
    {
        "nome":    "Monte Alegre",
        "pasta":   "data",
        "arquivo": "indicadores_monte_alegre_tratados.csv",
    },
    {
        "nome":    "Muaná",
        "pasta":   "data",
        "arquivo": "indicadores_muaná_tratados.csv",
    },
    {
        "nome":    "Novo Repartimento",
        "pasta":   "data",
        "arquivo": "indicadores_novo_repartimento_tratados.csv",
    },
    {
        "nome":    "Óbidos",
        "pasta":   "data",
        "arquivo": "indicadores_óbidos_tratados.csv",
    },
    {
        "nome":    "Oriximiná",
        "pasta":   "data",
        "arquivo": "indicadores_oriximiná_tratados.csv",
    },
    {
        "nome":    "Paragominas",
        "pasta":   "data",
        "arquivo": "indicadores_paragominas_tratados.csv",
    },
    {
        "nome":    "Parauapebas",
        "pasta":   "data",
        "arquivo": "indicadores_parauapebas_tratados.csv",
    },
    {
        "nome":    "Portel",
        "pasta":   "data",
        "arquivo": "indicadores_portel_tratados.csv",
    },
    {
        "nome":    "Redenção",
        "pasta":   "data",
        "arquivo": "indicadores_redenção_tratados.csv",
    },
    {
        "nome":    "Rondon do Pará",
        "pasta":   "data",
        "arquivo": "indicadores_rondon_do_pará_tratados.csv",
    },
    {
        "nome":    "Salinópolis",
        "pasta":   "data",
        "arquivo": "indicadores_salinópolis_tratados.csv",
    },
    {
        "nome":    "Santa Izabel do Pará",
        "pasta":   "data",
        "arquivo": "indicadores_santa_izabel_do_pará_tratados.csv",
    },
    {
        "nome":    "Santarém",
        "pasta":   "data",
        "arquivo": "indicadores_santarem_tratados.csv",
    },
    {
        "nome":    "São Félix do Xingu",
        "pasta":   "data",
        "arquivo": "indicadores_são_félix_do_xingu_tratados.csv",
    },
    {
        "nome":    "São Miguel do Guamá",
        "pasta":   "data",
        "arquivo": "indicadores_são_miguel_do_guamá_tratados.csv",
    },
    {
        "nome":    "Tailândia",
        "pasta":   "data",
        "arquivo": "indicadores_tailândia_tratados.csv",
    },
    {
        "nome":    "Tomé-Açu",
        "pasta":   "data",
        "arquivo": "indicadores_tomé_açú_tratados.csv",
    },
    {
        "nome":    "Tucuruí",
        "pasta":   "data",
        "arquivo": "indicadores_tucuruí_tratados.csv",
    },
    {
        "nome":    "Uruará",
        "pasta":   "data",
        "arquivo": "indicadores_uruará_tratados.csv",
    },
    {
        "nome":    "Vigia",
        "pasta":   "data",
        "arquivo": "indicadores_vigia_tratados.csv",
    },
    {
        "nome":    "Viseu",
        "pasta":   "data",
        "arquivo": "indicadores_viseu_tratados.csv",
    },
    {
        "nome":    "Xinguara",
        "pasta":   "data",
        "arquivo": "indicadores_xinguara_tratados.csv",
    },
    {
        "nome":    "Porto de Moz",
        "pasta":   "data",
        "arquivo": "indicadores_porto_de_moz_tratados.csv",
    },
    {
        "nome":    "Pacajá",
        "pasta":   "data",
        "arquivo": "indicadores_pacajá_tratados.csv",
    },
    {
        "nome":    "Tucumã",
        "pasta":   "data",
        "arquivo": "indicadores_tucumã_tratados.csv",
    },
    {
        "nome":    "Afuá",
        "pasta":   "data",
        "arquivo": "indicadores_afuá_tratados.csv",
    },
    {
        "nome":    "Ulianópolis",
        "pasta":   "data",
        "arquivo": "indicadores_ulianópolis_tratados.csv",
    },
    {
        "nome":    "Prainha",
        "pasta":   "data",
        "arquivo": "indicadores_prainha_tratados.csv",
    },
    {
        "nome":    "Igarapé-Açu",
        "pasta":   "data",
        "arquivo": "indicadores_igarapé-açu_tratados.csv",
    },
    {
        "nome":    "Jacundá",
        "pasta":   "data",
        "arquivo": "indicadores_jacundá_tratados.csv",
    },
    {
        "nome":    "Mãe do Rio",
        "pasta":   "data",
        "arquivo": "indicadores_mãe_do_rio_tratados.csv",
    },
    {
        "nome":    "Rurópolis",
        "pasta":   "data",
        "arquivo": "indicadores_rurópolis_tratados.csv",
    },
    {
        "nome":    "Novo Progresso",
        "pasta":   "data",
        "arquivo": "indicadores_novo_progresso_tratados.csv",
    },
    {
        "nome":    "Curralinho",
        "pasta":   "data",
        "arquivo": "indicadores_curralinho_tratados.csv",
    },
    {
        "nome":    "Oeiras do Pará",
        "pasta":   "data",
        "arquivo": "indicadores_oeiras_do_pará_tratados.csv",
    },
    {
        "nome":    "Almeirim",
        "pasta":   "data",
        "arquivo": "indicadores_almeirim_tratados.csv",
    },
    {
        "nome":    "Anapu",
        "pasta":   "data",
        "arquivo": "indicadores_anapu_tratados.csv",
    },
    {
        "nome":    "Ourilândia do Norte",
        "pasta":   "data",
        "arquivo": "indicadores_ourilândia_do_norte_tratados.csv",
    },
    {
        "nome":    "Bagre",
        "pasta":   "data",
        "arquivo": "indicadores_bagre_tratados.csv",
    },
    {
        "nome":    "Gurupá",
        "pasta":   "data",
        "arquivo": "indicadores_gurupá_tratados.csv",
    },
    {
        "nome":    "Irituia",
        "pasta":   "data",
        "arquivo": "indicadores_irituia_tratados.csv",
    },
    {
        "nome":    "São Domingos do Capim",
        "pasta":   "data",
        "arquivo": "indicadores_são_domingos_do_capim_tratados.csv",
    },
    {
        "nome":    "Limoeiro do Ajuru",
        "pasta":   "data",
        "arquivo": "indicadores_limoeiro_do_ajuru_tratados.csv",
    },
    {
        "nome":    "Tracuateua",
        "pasta":   "data",
        "arquivo": "indicadores_tracuateua_tratados.csv",
    },
    {
        "nome":    "Santana do Araguaia",
        "pasta":   "data",
        "arquivo": "indicadores_santana_do_araguaia_tratados.csv",
    },
    {
        "nome":    "Anajás",
        "pasta":   "data",
        "arquivo": "indicadores_anajás_tratados.csv",
    },
    {
        "nome":    "Melgaço",
        "pasta":   "data",
        "arquivo": "indicadores_melgaço_tratados.csv",
    },
    {
        "nome":    "Eldorado do Carajás",
        "pasta":   "data",
        "arquivo": "indicadores_eldorado_do_carajás_tratados.csv",
    },
    {
        "nome":    "Ipixuna do Pará",
        "pasta":   "data",
        "arquivo": "indicadores_ipixuna_do_pará_tratados.csv",
    },
    {
        "nome":    "Santo Antônio do Tauá",
        "pasta":   "data",
        "arquivo": "indicadores_santo_antônio_do_tauá_tratados.csv",
    },
    {
        "nome":    "Mocajuba",
        "pasta":   "data",
        "arquivo": "indicadores_mocajuba_tratados.csv",
    },
    {
        "nome":    "Medicilândia",
        "pasta":   "data",
        "arquivo": "indicadores_medicilândia_tratados.csv",
    },
    {
        "nome":    "Concórdia do Pará",
        "pasta":   "data",
        "arquivo": "indicadores_concórdia_do_pará_tratados.csv",
    },
    {
        "nome":    "Marapanim",
        "pasta":   "data",
        "arquivo": "indicadores_marapanim_tratados.csv",
    },
    {
        "nome":    "São Sebastião da Boa Vista",
        "pasta":   "data",
        "arquivo": "indicadores_são_sebastião_da_boa_vista_tratados.csv",
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
# IDENTIDADE VISUAL - REFINADA (SCHOLARIS DARK THEME)
# ═══════════════════════════════════════════════════════════════════════════

COR_PRIMARIA   = "#0C1222"   # Fundo escuro sidebar/main
COR_PRIMARIA_2 = "#152336"   # Fundo secundário
COR_MARCA      = "#38BDF8"   # Azul neon (destaque principal)
COR_ACENTO     = "#818CF8"   # Roxo/Azul claro para projeções
COR_SUCESSO    = "#10B981"
COR_NEUTRA     = "#94A3B8"
COR_TEXTO      = "#E2E8F0"
COR_FUNDO_APP  = "#080D19"   # Fundo principal muito escuro
COR_FUNDO_CARD = "rgba(15, 23, 42, 0.6)"
COR_BORDA      = "#1E293B"

PALETA_COMPARATIVO = [
    "#38BDF8", "#818CF8", "#A855F7", "#F472B6",
    "#34D399", "#FBBF24", "#F87171", "#60A5FA",
    "#C084FC", "#2DD4BF", "#FB923C", "#9CA3AF",
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


@st.cache_data(show_spinner=False)
def _logo_base64() -> str | None:
    """Lê o logo em assets/ e devolve como base64 para embutir no HTML do painel."""
    # RESOLUÇÃO ROBUSTA DE CAMINHO: Tenta várias opções para encontrar o arquivo
    caminho_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOGO_ARQUIVO)
    caminho_repo = os.path.join(_raiz_repo(), LOGO_ARQUIVO)
    caminho_cwd = os.path.join(os.getcwd(), LOGO_ARQUIVO)
    
    caminho_final = None
    if os.path.exists(caminho_script):
        caminho_final = caminho_script
    elif os.path.exists(caminho_repo):
        caminho_final = caminho_repo
    elif os.path.exists(caminho_cwd):
        caminho_final = caminho_cwd
        
    if not caminho_final:
        return None
        
    with open(caminho_final, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _logo_mime() -> str:
    ext = os.path.splitext(LOGO_ARQUIVO)[1].lower()
    return {"png": "png", "jpg": "jpeg", "jpeg": "jpeg", "webp": "webp"}.get(ext.strip("."), "png")


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
# GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════

_FONTE = dict(family="Inter, -apple-system, sans-serif", color=COR_TEXTO)


def fig_serie(titulo: str, anos: list[int], valores: list[float],
              ylabel: str, municipio: str, ativacao: str, solver: str,
              vals_proj: list[float]) -> go.Figure:
    anos_curva, vals_curva = _curva_mlp(anos, valores, ativacao, solver)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=anos_curva, y=vals_curva,
        mode="lines", name=f"Curva MLP ({ativacao}/{solver})",
        line=dict(color=COR_ACENTO, width=2, dash="solid"), opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=anos, y=valores,
        mode="lines+markers", name="Censos IBGE",
        line=dict(color=COR_MARCA, width=3),
        marker=dict(size=11, color=COR_MARCA, line=dict(width=2, color="#080D19")),
    ))
    fig.add_trace(go.Scatter(
        x=ANOS_PROJECAO, y=vals_proj,
        mode="markers+text", name="Projeção",
        text=[f"{v:,.1f}" for v in vals_proj],
        textposition="top center",
        marker=dict(size=15, color="#A855F7", symbol="star",
                    line=dict(width=1.5, color="#E2E8F0")),
    ))

    fig.update_layout(
        template="plotly_dark", height=430,
        font=_FONTE,
        title=dict(text=f"<b>{titulo}</b> — {municipio}", x=0.01,
                   font=dict(size=17, color=COR_TEXTO)),
        xaxis_title="Ano",
        yaxis_title=ylabel,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0, font=dict(size=12)),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=64, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor=COR_BORDA, zeroline=False)
    fig.update_yaxes(gridcolor=COR_BORDA, zeroline=False)
    if max(valores) > 1000:
        fig.update_yaxes(tickformat=",")
    return fig


def fig_barras_todos(df: pd.DataFrame, municipio: str) -> go.Figure:
    fig = px.bar(
        df, x="indicador_nome", y="valor", color="ano",
        barmode="group", text_auto=True,
        title=f"Visão geral — {municipio}",
        color_continuous_scale=[COR_ACENTO, COR_MARCA],
        labels={"indicador_nome": "Indicador", "valor": "Valor", "ano": "Ano"},
    )
    fig.update_layout(
        template="plotly_dark", height=500,
        font=_FONTE,
        title=dict(font=dict(size=17, color=COR_TEXTO)),
        xaxis_tickangle=-35,
        margin=dict(l=40, r=20, t=60, b=120),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor=COR_BORDA)
    fig.update_yaxes(gridcolor=COR_BORDA)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ESTILO GLOBAL
# ═══════════════════════════════════════════════════════════════════════════

def _css() -> None:
    st.markdown(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

          html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}

          /* Fundo principal escuro com leve gradiente radial */
          .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
              background-color: {COR_FUNDO_APP} !important;
              background-image: radial-gradient(circle at 50% 0%, {COR_PRIMARIA_2} 0%, transparent 60%);
              color: {COR_TEXTO};
          }}
          [data-testid="stHeader"] {{ background: transparent !important; }}
          .block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1240px; }}

          h1, h2, h3, h4, p, span, label, li,
          [data-testid="stMarkdownContainer"] {{ color: {COR_TEXTO} !important; }}

          /* ---------- Cabeçalho de página centralizado ---------- */
          .painel-header {{
              text-align: center;
              padding: 2.5rem 1rem 1.5rem;
              margin-bottom: 2rem;
              position: relative;
          }}
          .painel-header-logo {{
              height: 55px;
              margin-bottom: 1rem;
              filter: drop-shadow(0 0 12px rgba(56, 189, 248, 0.4));
          }}
          .painel-header h2 {{
              font-size: 2.2rem;
              font-weight: 700;
              margin-bottom: 0.5rem;
              background: linear-gradient(90deg, #FFFFFF, {COR_MARCA});
              -webkit-background-clip: text;
              -webkit-text-fill-color: transparent;
              letter-spacing: 0.02em;
          }}
          .painel-header p {{
              color: {COR_NEUTRA} !important;
              font-size: 1.05rem;
              max-width: 800px;
              margin: 0 auto;
              line-height: 1.5;
          }}

          .badge {{
              display: inline-block;
              background: rgba(56, 189, 248, 0.1);
              color: {COR_MARCA} !important;
              padding: 0.25rem 0.8rem;
              border-radius: 999px;
              font-size: 0.75rem;
              font-weight: 600;
              letter-spacing: 0.1em;
              text-transform: uppercase;
              border: 1px solid rgba(56, 189, 248, 0.3);
              margin-bottom: 1rem;
              box-shadow: 0 0 10px rgba(56, 189, 248, 0.15);
          }}
          
          .badge-auto {{
              display: inline-block;
              background: rgba(16, 185, 129, 0.15);
              color: #34D399 !important;
              padding: 0.14rem 0.55rem;
              border-radius: 999px;
              font-size: 0.72rem;
              font-weight: 600;
              margin-left: 0.45rem;
              border: 1px solid rgba(16, 185, 129, 0.3);
          }}
          .badge-fixo {{
              display: inline-block;
              background: rgba(148, 163, 184, 0.15);
              color: #94A3B8 !important;
              padding: 0.14rem 0.55rem;
              border-radius: 999px;
              font-size: 0.72rem;
              font-weight: 500;
              margin-left: 0.45rem;
              border: 1px solid rgba(148, 163, 184, 0.3);
          }}

          .dado-ausente {{
              background: rgba(15, 23, 42, 0.6);
              border-left: 4px solid #F59E0B;
              padding: 0.9rem 1.1rem;
              border-radius: 8px;
              color: {COR_TEXTO} !important;
              border: 1px solid {COR_BORDA};
          }}
          .dado-ausente code {{ color: #FBBF24; background: rgba(0,0,0,0.3); padding: 2px 4px; border-radius: 4px; }}

          /* ---------- Métricas / Cards ---------- */
          div[data-testid="stMetric"] {{
              background: {COR_FUNDO_CARD} !important;
              border: 1px solid {COR_BORDA} !important;
              border-radius: 12px;
              padding: 1rem !important;
              box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
              backdrop-filter: blur(10px);
          }}
          div[data-testid="stMetricLabel"] {{ color: {COR_NEUTRA} !important; font-weight: 500; }}
          div[data-testid="stMetricValue"] {{ color: {COR_MARCA} !important; font-weight: 700; }}

          /* ---------- Sidebar ---------- */
          section[data-testid="stSidebar"] {{
              background-color: {COR_PRIMARIA} !important;
              border-right: 1px solid {COR_BORDA} !important;
          }}
          section[data-testid="stSidebar"] h3 {{ color: #FFFFFF !important; font-weight: 600; font-size: 1.1rem; letter-spacing: 0.05em; }}
          .sidebar-logo {{ display: flex; justify-content: center; padding: 1rem 0 1.5rem; }}
          .sidebar-logo img {{ max-width: 140px; filter: drop-shadow(0 0 8px rgba(56,189,248,0.3)); }}

          /* ---------- Inputs (radio, texto, selectbox) ---------- */
          [data-baseweb="input"], [data-baseweb="select"] > div {{
              background-color: rgba(15, 23, 42, 0.8) !important;
              border: 1px solid {COR_BORDA} !important;
              border-radius: 8px !important;
              color: {COR_TEXTO} !important;
          }}
          [data-baseweb="base-input"] input {{
              color: {COR_TEXTO} !important;
          }}
          div[role="radiogroup"] label {{
              border-radius: 8px;
              padding: 0.5rem 0.8rem;
              margin-bottom: 0.2rem;
              transition: all 0.2s ease;
              background-color: transparent;
              border: 1px solid transparent;
          }}
          div[role="radiogroup"] label:hover {{
              background-color: rgba(56, 189, 248, 0.1);
              border: 1px solid rgba(56, 189, 248, 0.3);
          }}
          div[role="radiogroup"] label[data-baseweb="radio"] div:first-child {{
              background-color: transparent !important;
              border-color: {COR_MARCA} !important;
          }}
          
          /* ---------- Botões ---------- */
          .stButton > button {{
              background: rgba(56, 189, 248, 0.1) !important;
              border: 1px solid rgba(56, 189, 248, 0.3) !important;
              color: {COR_MARCA} !important;
              border-radius: 8px !important;
              font-weight: 500 !important;
              transition: all 0.2s ease !important;
          }}
          .stButton > button:hover {{
              background: rgba(56, 189, 248, 0.2) !important;
              border-color: {COR_MARCA} !important;
              box-shadow: 0 0 15px rgba(56, 189, 248, 0.2) !important;
              color: #FFF !important;
          }}

          /* ---------- Expanders / cartões ---------- */
          div[data-testid="stExpander"] {{
              background: {COR_FUNDO_CARD} !important;
              border: 1px solid {COR_BORDA} !important;
              border-radius: 12px !important;
          }}
          div[data-testid="stExpander"] summary {{ font-weight: 600; color: #FFFFFF !important; }}

          hr {{ border-color: {COR_BORDA} !important; margin: 1.5rem 0; opacity: 0.7; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(badge: str, titulo: str, descricao: str) -> None:
    logo_b64 = _logo_base64()
    logo_html = (
        f'<img class="painel-header-logo" '
        f'src="data:image/{_logo_mime()};base64,{logo_b64}">'
        if logo_b64 else ""
    )
    st.markdown(
        f"""
        <div class="painel-header">
          {logo_html}
          <br>
          <span class="badge">{badge}</span>
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
        "MUNICÍPIO",
        nome,
        "Dados do IBGE (Censos 1991–2022) com projeções MLP para "
        f"{' e '.join(str(a) for a in ANOS_PROJECAO)}, usando a "
        "ativação/solver escolhida automaticamente por indicador "
        "(validação leave-one-out).",
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
            f"Projeção ({ylabel})": [round(v, 2) for v in vals_proj],
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


# ═══════════════════════════════════════════════════════════════════════════
# NAVEGAÇÃO LATERAL
# ═══════════════════════════════════════════════════════════════════════════

def _sidebar_navegacao(nomes_municipios: list[str]) -> str:
    """Sidebar com logo e barra de busca sempre limpa."""
    logo_b64 = _logo_base64()
    if logo_b64:
        st.sidebar.markdown(
            f'<div class="sidebar-logo"><img src="data:image/{_logo_mime()};base64,{logo_b64}"></div>',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown(f"### ❖ {TITULO_PAINEL}")

    if st.sidebar.button("🔄 Recarregar dados", help="Limpa o cache após subir novos dados no GitHub",
                         use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.divider()

    # Define a página inicial se for o primeiro acesso
    if "pagina_atual" not in st.session_state:
        st.session_state["pagina_atual"] = nomes_municipios[0]

    lista_opcoes = list(nomes_municipios)

    # Callback: O que acontece ao escolher algo na barra de busca
    def ao_buscar():
        escolha = st.session_state.caixa_busca
        if escolha: # Se o usuário escolheu algo (não apenas limpou a caixa)
            st.session_state["pagina_atual"] = escolha
            # O segredo: forçamos a caixa de busca a voltar a ficar vazia após a navegação!
            st.session_state.caixa_busca = None

    # O selectbox agora funciona puramente como um "Spotlight" / "Jump to"
    st.sidebar.selectbox(
        "Navegar",
        options=lista_opcoes,
        index=None, # Define o estado padrão como vazio
        placeholder="🔎 Clique para buscar...",
        key="caixa_busca",
        on_change=ao_buscar, # Executa a função acima ao clicar em um município
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "Projeções via MLP (sklearn), modelo por indicador escolhido via LOOCV."
    )
    st.sidebar.info(
        "Os dados têm como base o Censo do IBGE, o SIDRA, o Panorama do "
        "Censo e o DATASUS. Pode haver margem de "
        "erro para mais ou para menos.",
        icon="ℹ️",
    )

    return st.session_state["pagina_atual"]


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title=TITULO_PAINEL,
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="❖",
    )
    _css()

    nomes_municipios = [cfg["nome"] for cfg in MUNICIPIOS]
    pagina = _sidebar_navegacao(nomes_municipios)

    cfg = next((c for c in MUNICIPIOS if c["nome"] == pagina), MUNICIPIOS[0])
    render_municipio(cfg)


if __name__ == "__main__":
    main()
