# -*- coding: utf-8 -*-
"""
app.py — Painel Streamlit TCC (coringa multi-município)
Versão com Autenticação (streamlit-authenticator) e Controle de Acesso (Admin vs Viewer).
"""

from __future__ import annotations

import base64
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit_authenticator as stauth

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║              BLOCO DE CONFIGURAÇÃO — EDITE APENAS AQUI                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

TITULO_PAINEL = "Censo IBGE — Projeções Municipais"
LOGO_ARQUIVO = "assets/logo_isaci_neon.webp"

MUNICIPIOS = [
    {"nome": "Castanhal", "pasta": "data", "arquivo": "indicadores_castanhal_tratados.csv"},
    {"nome": "Belém", "pasta": "data", "arquivo": "indicadores_belem_tratados.csv"},
    {"nome": "Ananindeua", "pasta": "data", "arquivo": "indicadores_ananindeua_tratados.csv"},
    {"nome": "Santarém", "pasta": "data", "arquivo": "indicadores_santarem_tratados.csv"},
    {"nome": "Parauapebas", "pasta": "data", "arquivo": "indicadores_parauapebas_tratados.csv"},
    {"nome": "Abaetetuba", "pasta": "data", "arquivo": "indicadores_abaetetuba_tratados.csv"},
    {"nome": "Barcarena", "pasta": "data", "arquivo": "indicadores_barcarena_tratados.csv"},
    {"nome": "Cametá", "pasta": "data", "arquivo": "indicadores_cametá_tratados.csv"},
    {"nome": "Marabá", "pasta": "data", "arquivo": "indicadores_maraba_tratados.csv"},
    {"nome": "Altamira", "pasta": "data", "arquivo": "indicadores_altamira_tratados.csv"},
    {"nome": "Itaituba", "pasta": "data", "arquivo": "indicadores_itaituba_tratados.csv"},
    {"nome": "Bragança", "pasta": "data", "arquivo": "indicadores_bragança_tratados.csv"},
]

NIVEIS_ACIMA_PARA_DADOS = 1
ANOS_PROJECAO = [2030, 2040]

ATIVACAO_FALLBACK = "relu"
SOLVER_FALLBACK   = "lbfgs"
HIDDEN_LAYERS     = (10, 10)
RANDOM_STATE      = 42
MAX_ITER          = 500

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     FIM DO BLOCO DE CONFIGURAÇÃO                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

COR_PRIMARIA   = "#134E5E"
COR_PRIMARIA_2 = "#0E7490"
COR_MARCA      = "#8BC53F"
COR_ACENTO     = "#F59E0B"
COR_SUCESSO    = "#2E9E4D"
COR_NEUTRA     = "#64748B"
COR_TEXTO      = "#1E293B"
COR_FUNDO_APP  = "#F4F7F8"
COR_FUNDO_CARD = "#FFFFFF"
COR_BORDA      = "#E2E8F0"

PALETA_COMPARATIVO = [
    "#134E5E", "#F59E0B", "#7C3AED", "#DC2626",
    "#0891B2", "#8BC53F", "#DB2777", "#4338CA",
    "#EA580C", "#0D9488", "#9333EA", "#B45309",
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
    caminho = os.path.join(_raiz_repo(), LOGO_ARQUIVO)
    if not os.path.exists(caminho):
        return None
    with open(caminho, "rb") as f:
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
    if "ativacao" not in df.columns: df["ativacao"] = ATIVACAO_FALLBACK
    if "solver" not in df.columns: df["solver"] = SOLVER_FALLBACK
    if "auto_selecionado" not in df.columns: df["auto_selecionado"] = False
    if "loocv_mae" not in df.columns: df["loocv_mae"] = np.nan
    if "indicador_id" not in df.columns: df["indicador_id"] = df.get("indicador_nome", "")
    return df

@st.cache_data(show_spinner=False, ttl="24h")
def carregar_projecoes(pasta: str, arquivo_historico: str) -> pd.DataFrame | None:
    nome_proj = _arquivo_projecoes(arquivo_historico)
    if not nome_proj: return None
    caminho = os.path.join(_raiz_repo(), pasta, nome_proj)
    if not os.path.exists(caminho): return None
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
        hidden_layer_sizes=HIDDEN_LAYERS, activation=ativacao, solver=solver,
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

def _curva_mlp(anos: list[int], valores: list[float],
               ativacao: str, solver: str) -> tuple[list[int], list[float]]:
    return _curva_mlp_cached(tuple(anos), tuple(valores), ativacao, solver)

def _projecao_indicador(ind_id: str, anos: list[int], valores: list[float],
                        ativacao: str, solver: str, df_proj_precalc: pd.DataFrame | None) -> list[float]:
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
        x=anos_curva, y=vals_curva, mode="lines", name=f"Curva MLP ({ativacao}/{solver})",
        line=dict(color=COR_PRIMARIA_2, width=2, dash="solid"), opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=anos, y=valores, mode="lines+markers", name="Censos IBGE",
        line=dict(color=COR_PRIMARIA, width=3),
        marker=dict(size=11, color=COR_PRIMARIA, line=dict(width=2, color="white")),
    ))
    fig.add_trace(go.Scatter(
        x=ANOS_PROJECAO, y=vals_proj, mode="markers+text", name="Projeção",
        text=[f"{v:,.1f}" for v in vals_proj], textposition="top center",
        marker=dict(size=15, color=COR_ACENTO, symbol="star", line=dict(width=1.5, color="#7C2D12")),
    ))
    fig.update_layout(
        template="plotly_white", height=430, font=_FONTE,
        title=dict(text=f"<b>{titulo}</b> — {municipio}", x=0.01, font=dict(size=17, color=COR_PRIMARIA)),
        xaxis_title="Ano", yaxis_title=ylabel,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0, font=dict(size=12)),
        hovermode="x unified", margin=dict(l=50, r=20, t=64, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#EEF2F6", zeroline=False)
    fig.update_yaxes(gridcolor="#EEF2F6", zeroline=False)
    if max(valores) > 1000: fig.update_yaxes(tickformat=",")
    return fig

def fig_barras_todos(df: pd.DataFrame, municipio: str) -> go.Figure:
    fig = px.bar(
        df, x="indicador_nome", y="valor", color="ano", barmode="group", text_auto=True,
        title=f"Visão geral — {municipio}", color_continuous_scale=[COR_PRIMARIA_2, COR_ACENTO],
        labels={"indicador_nome": "Indicador", "valor": "Valor", "ano": "Ano"},
    )
    fig.update_layout(
        template="plotly_white", height=500, font=_FONTE,
        title=dict(font=dict(size=17, color=COR_PRIMARIA)),
        xaxis_tickangle=-35, margin=dict(l=40, r=20, t=60, b=120),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#EEF2F6")
    fig.update_yaxes(gridcolor="#EEF2F6")
    return fig

def fig_comparativo(dfs: dict[str, pd.DataFrame], dfs_proj: dict, ind_escolhido: str) -> tuple[go.Figure, str, bool]:
    fig = go.Figure()
    cores = PALETA_COMPARATIVO
    tem_dados = False
    ylabel = "Valor"

    for i, (mun, df) in enumerate(dfs.items()):
        sub = df[df["indicador_nome"] == ind_escolhido].sort_values("ano")
        if sub.empty: continue
        anos = sub["ano"].astype(int).tolist()
        vals = sub["valor"].tolist()
        if len(anos) < 2 or any(np.isnan(v) for v in vals): continue

        ind_id   = sub["indicador_id"].iloc[0]
        ativacao = sub["ativacao"].iloc[0]
        solver   = sub["solver"].iloc[0]
        ylabel   = sub["unidade_medida"].iloc[0] if "unidade_medida" in sub.columns else "Valor"
        cor      = cores[i % len(cores)]

        fig.add_trace(go.Scatter(
            x=anos, y=vals, mode="lines+markers", name=mun, legendgroup=mun, showlegend=True,
            line=dict(color=cor, width=3), marker=dict(size=8, line=dict(width=1.5, color="white")),
            hovertemplate=f"<b>{mun}</b> — %{{x}}: %{{y:,.2f}}<extra></extra>",
        ))

        vals_proj = _projecao_indicador(ind_id, anos, vals, ativacao, solver, dfs_proj.get(mun))
        fig.add_trace(go.Scatter(
            x=ANOS_PROJECAO, y=vals_proj, mode="markers", name=mun, legendgroup=mun, showlegend=False,
            marker=dict(size=12, symbol="star", color=cor, line=dict(width=1, color="#334155")),
            hovertemplate=(f"<b>{mun}</b> — projeção %{{x}}: %{{y:,.2f}} ({ativacao}/{solver})<extra></extra>"),
        ))
        tem_dados = True

    fig.update_layout(
        template="plotly_white", height=500, font=_FONTE,
        title=dict(text=f"<b>{ind_escolhido}</b>", x=0.01, font=dict(size=17, color=COR_PRIMARIA)),
        xaxis_title="Ano", yaxis_title=ylabel,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.015, font=dict(size=11.5),
                    bgcolor="rgba(255,255,255,0.7)", bordercolor=COR_BORDA, borderwidth=1, tracegroupgap=2),
        hovermode="closest", margin=dict(l=50, r=170, t=60, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#EEF2F6")
    fig.update_yaxes(gridcolor="#EEF2F6")
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
            background-color: {COR_FUNDO_APP} !important;
          }}
          [data-testid="stHeader"] {{ background: transparent !important; }}
          .block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1240px; }}

          h1, h2, h3, h4, p, span, label, li,
          [data-testid="stMarkdownContainer"] {{ color: {COR_TEXTO}; }}

          /* ---------- Cabeçalho de página ---------- */
          .painel-header {{
            position: relative;
            background: linear-gradient(125deg, {COR_PRIMARIA} 0%, {COR_PRIMARIA_2} 55%, #0891B2 100%);
            padding: 1.7rem 2rem;
            border-radius: 18px;
            margin-bottom: 1.4rem;
            box-shadow: 0 10px 28px -10px rgba(19, 78, 94, 0.5);
            overflow: hidden;
          }}
          .painel-header::after {{
            content: "";
            position: absolute; inset: 0;
            background: radial-gradient(circle at 85% -20%, rgba(139,197,63,0.35), transparent 55%);
            pointer-events: none;
          }}
          .painel-header h2 {{
            margin: 0.3rem 0 0.2rem; color: #FFFFFF !important; font-weight: 800;
            letter-spacing: -0.01em; font-size: 1.7rem;
          }}
          .painel-header p {{
            margin: 0; color: #E0F2FE !important; font-size: 0.93rem; max-width: 760px;
            line-height: 1.5; position: relative; z-index: 1;
          }}
          .painel-header-logo {{
            position: absolute; top: 1.2rem; right: 1.6rem; height: 30px; opacity: 0.95; z-index: 1;
          }}

          .badge {{
            display: inline-block; background: rgba(255,255,255,0.16); color: #FFFFFF !important;
            padding: 0.2rem 0.7rem; border-radius: 999px; font-size: 0.73rem; font-weight: 700;
            letter-spacing: 0.04em; text-transform: uppercase; border: 1px solid rgba(255,255,255,0.35);
            position: relative; z-index: 1;
          }}
          .badge-auto {{
            display: inline-block; background: {COR_SUCESSO}; color: white !important;
            padding: 0.14rem 0.55rem; border-radius: 999px; font-size: 0.72rem; font-weight: 500; margin-left: 0.45rem;
          }}
          .badge-fixo {{
            display: inline-block; background: {COR_NEUTRA}; color: white !important;
            padding: 0.14rem 0.55rem; border-radius: 999px; font-size: 0.72rem; font-weight: 500; margin-left: 0.45rem;
          }}

          .dado-ausente {{
            background: #FFFBEB; border-left: 4px solid {COR_ACENTO}; padding: 0.9rem 1.1rem;
            border-radius: 8px; color: {COR_TEXTO} !important;
          }}
          .dado-ausente code {{ color: #92400E; }}

          /* ---------- Métricas ---------- */
          div[data-testid="stMetric"] {{
            background: {COR_FUNDO_CARD}; border: 1px solid {COR_BORDA}; border-radius: 12px;
            padding: 0.8rem 1rem 0.6rem; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
          }}
          div[data-testid="stMetricLabel"] {{ color: {COR_NEUTRA} !important; font-weight: 500; }}
          div[data-testid="stMetricValue"] {{ color: {COR_PRIMARIA} !important; }}

          /* ---------- Sidebar ---------- */
          section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #FFFFFF 0%, #F4F7F8 100%) !important;
            border-right: 1px solid {COR_BORDA};
          }}
          section[data-testid="stSidebar"] * {{ color: {COR_TEXTO}; }}
          section[data-testid="stSidebar"] h3 {{ color: {COR_PRIMARIA} !important; font-weight: 800; }}
          .sidebar-logo {{ display: flex; justify-content: center; padding: 0.4rem 0 1rem; }}
          .sidebar-logo img {{ max-width: 168px; }}

          /* ---------- Inputs (radio, texto, selectbox) ---------- */
          [data-baseweb="input"], [data-baseweb="select"] > div {{
            background-color: white !important; border-radius: 10px !important; border-color: {COR_BORDA} !important;
          }}
          div[role="radiogroup"] label {{
            border-radius: 10px; padding: 0.3rem 0.5rem; margin-bottom: 0.1rem; transition: background-color 0.15s ease;
          }}
          div[role="radiogroup"] label:hover {{ background-color: #E6F4EA; }}
          div[role="radiogroup"] label[data-baseweb="radio"] div:first-child {{
            border-color: {COR_MARCA} !important;
          }}
          .stButton > button {{
            border-radius: 10px !important; border: 1px solid {COR_BORDA} !important; font-weight: 600 !important;
          }}
          .stButton > button:hover {{ border-color: {COR_PRIMARIA} !important; color: {COR_PRIMARIA} !important; }}

          /* ---------- Expanders / cartões ---------- */
          div[data-testid="stExpander"] {{
            border: 1px solid {COR_BORDA} !important; border-radius: 12px !important;
            background: {COR_FUNDO_CARD} !important; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
          }}
          div[data-testid="stExpander"] summary {{ font-weight: 600; }}
          hr {{ margin: 1.1rem 0; opacity: 0.5; }}
          
          /* Estilos para a tela de Login */
          .stTextInput > div > div > input {{ background-color: white !important; }}
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
        "Município", nome,
        "Dados do IBGE (Censos 1991–2022) com projeções MLP para "
        f"{' e '.join(str(a) for a in ANOS_PROJECAO)}, usando a "
        "ativação/solver escolhida automaticamente por indicador."
    )

    if df is None:
        st.markdown(
            f"""
            <div class="dado-ausente">
              <strong>⚠️ Arquivo não encontrado:</strong> <code>{cfg['pasta']}/{cfg['arquivo']}</code>
            </div>
            """, unsafe_allow_html=True
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

    st.divider()

    with st.expander("📊 Visão geral — todos os indicadores", expanded=False):
        st.plotly_chart(fig_barras_todos(df, nome), use_container_width=True)

    st.divider()

    with st.expander("🗂️ Tabela de dados brutos", expanded=False):
        st.dataframe(df.sort_values(["indicador_nome", "ano"]), use_container_width=True, hide_index=True)
        
        # -------------------------------------------------------------
        # PRINCÍPIO DO MENOR PRIVILÉGIO: Checa o cargo do usuário logado
        # -------------------------------------------------------------
        if st.session_state.get("user_role") == "admin":
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Baixar Base CSV (Exclusivo Admin)",
                data=csv,
                file_name=cfg["arquivo"],
                mime="text/csv",
            )
        else:
            st.info("🔒 O download da base de dados brutos é uma funcionalidade restrita a administradores.")

    st.divider()
    st.subheader("Séries históricas e projeções MLP")

    indicadores = sorted(df["indicador_nome"].unique())
    escolha = st.selectbox("Indicador", options=indicadores + ["(Exibir Todos)"], key=f"sel_{nome}")

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

        if len(anos) < 2 or any(np.isnan(v) for v in vals): return

        badge_modelo = (
            f'<span class="badge-auto">auto — LOOCV MAE={loocv:,.2f}</span>' if auto and pd.notna(loocv)
            else f'<span class="badge-fixo">modelo fixo</span>'
        )
        st.markdown(f"**{ativacao} / {solver}** {badge_modelo}", unsafe_allow_html=True)

        vals_proj = _projecao_indicador(ind_id, anos, vals, ativacao, solver, df_proj_precalc)
        st.plotly_chart(fig_serie(ind_nome, anos, vals, ylabel, nome, ativacao, solver, vals_proj), use_container_width=True)

        df_proj_tabela = pd.DataFrame({"Ano": ANOS_PROJECAO, f"Projeção MLP ({ylabel})": [round(v, 2) for v in vals_proj]})
        col_tab, col_esp = st.columns([1, 2])
        col_tab.caption(f"Grupo: **{grupo}** | Censos na série: {anos}")
        col_tab.dataframe(df_proj_tabela, hide_index=True, use_container_width=True)
        st.divider()

    if escolha == "(Exibir Todos)":
        for ind in indicadores: _render_grafico_indicador(ind)
    else:
        _render_grafico_indicador(escolha)


def render_comparativo(municipios_carregados: list[dict]) -> None:
    _render_header("Análise cruzada", "Comparativo entre municípios", "Selecione um indicador para visualizar a evolução histórica lado a lado.")
    todos_indicadores: set[str] = set()
    dfs, dfs_proj = {}, {}
    
    for cfg in municipios_carregados:
        df = carregar_dados(cfg["pasta"], cfg["arquivo"])
        if df is not None:
            dfs[cfg["nome"]] = df
            dfs_proj[cfg["nome"]] = carregar_projecoes(cfg["pasta"], cfg["arquivo"])
            todos_indicadores.update(df["indicador_nome"].unique())

    if not dfs: return

    ind_escolhido = st.selectbox("Indicador para comparar", sorted(todos_indicadores), key="sel_comparativo")
    fig, ylabel, tem_dados = fig_comparativo(dfs, dfs_proj, ind_escolhido)

    if tem_dados:
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# NAVEGAÇÃO LATERAL
# ═══════════════════════════════════════════════════════════════════════════

def _sidebar_navegacao(nomes_municipios: list[str]) -> str:
    logo_b64 = _logo_base64()
    if logo_b64:
        st.sidebar.markdown(f'<div class="sidebar-logo"><img src="data:image/{_logo_mime()};base64,{logo_b64}"></div>', unsafe_allow_html=True)

    st.sidebar.markdown(f"### 📊 {TITULO_PAINEL}")

    if st.sidebar.button("🔄 Recarregar dados", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.divider()
    busca = st.sidebar.text_input("Buscar município", placeholder="🔎 Digite para filtrar…", label_visibility="collapsed")
    opcoes = [n for n in nomes_municipios if busca.strip().lower() in n.lower()] if busca else list(nomes_municipios)

    if "pagina_atual" not in st.session_state: st.session_state["pagina_atual"] = nomes_municipios[0]

    pagina_atual = st.session_state["pagina_atual"]
    lista_radio = opcoes + ["🔀 Comparativo entre municípios"]
    default_idx = lista_radio.index(pagina_atual) if pagina_atual in lista_radio else 0

    escolha = st.sidebar.radio("Município", options=lista_radio, index=default_idx, label_visibility="collapsed")
    st.session_state["pagina_atual"] = "Comparativo" if escolha == "🔀 Comparativo entre municípios" else escolha
    return st.session_state["pagina_atual"]


# ═══════════════════════════════════════════════════════════════════════════
# MAIN - COM PROTEÇÃO DE LOGIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title=TITULO_PAINEL,
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="📊",
    )
    _css()

    if "auth" not in st.secrets:
        st.warning("⚠️ Configurações de Login não encontradas. Configure as 'Secrets' no painel do Streamlit Cloud.")
        st.stop()

    # Transforma as secrets do Streamlit em um dicionário Python compreensível pela biblioteca
    auth_config = json.loads(json.dumps(st.secrets["auth"].to_dict()))

    # Inicia a tela de login nativa do authenticator
    authenticator = stauth.Authenticate(
        auth_config['credentials'],
        auth_config['cookie']['name'],
        auth_config['cookie']['key'],
        auth_config['cookie']['expiry_days'],
    )

    try:
        name, authentication_status, username = authenticator.login("main")
    except Exception:
        name, authentication_status, username = authenticator.login("Acesso Restrito - NDI", "main")

    if authentication_status is False:
        st.error("❌ Usuário ou senha incorretos. Tente novamente.")
    
    elif authentication_status is None:
        st.info("Insira suas credenciais para acessar os indicadores.")
    
    elif authentication_status is True:
        # ---- SE LOGOU COM SUCESSO ----
        
        # Lê o cargo (role) configurado no painel da nuvem. Se não existir, vira "viewer" por segurança.
        role = auth_config['credentials']['usernames'][username].get('role', 'viewer')
        st.session_state["user_role"] = role
        
        # Mostra quem está logado e o botão de sair na barra lateral
        st.sidebar.markdown(f"Logado como: **{name}**")
        
        # Exibe uma etiqueta amarela ou cinza dependendo do perfil
        cor_tag = "#F59E0B" if role == "admin" else "#64748B"
        st.sidebar.markdown(f"<span style='background:{cor_tag}; color:white; padding:2px 8px; border-radius:10px; font-size:12px; font-weight:bold;'>PERFIL: {role.upper()}</span>", unsafe_allow_html=True)
        
        authenticator.logout("Sair", "sidebar")
        st.sidebar.divider()

        # Renderização do Dashboard real
        nomes_municipios = [cfg["nome"] for cfg in MUNICIPIOS]
        pagina = _sidebar_navegacao(nomes_municipios)

        if pagina == "Comparativo":
            render_comparativo(MUNICIPIOS)
        else:
            cfg = next((c for c in MUNICIPIOS if c["nome"] == pagina), MUNICIPIOS[0])
            render_municipio(cfg)


if __name__ == "__main__":
    main()
