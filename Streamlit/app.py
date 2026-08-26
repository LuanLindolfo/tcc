# -*- coding: utf-8 -*-
"""
app.py — Painel Streamlit TCC (coringa multi-município)
Versão com Navegação Intuitiva por Sub-abas, Filtros por Grupo e Caching de ML.
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
        "nome":    "Marabá",
        "pasta":   "data",
        "arquivo": "indicadores_maraba_tratados.csv",
    },
]

NIVEIS_ACIMA_PARA_DADOS = 1
ANOS_PROJECAO = [2030, 2040]

ATIVACAO_FALLBACK = "relu"
SOLVER_FALLBACK   = "lbfgs"
HIDDEN_LAYERS      = (10, 10)
RANDOM_STATE       = 42
MAX_ITER          = 500

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     FIM DO BLOCO DE CONFIGURAÇÃO                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS & CACHING DE DADOS / ML
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

def fig_serie(titulo: str, anos: list[int], valores: list[float],
              ylabel: str, municipio: str, ativacao: str, solver: str,
              vals_proj: list[float]) -> go.Figure:
    anos_curva, vals_curva = _curva_mlp(anos, valores, ativacao, solver)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=anos_curva, y=vals_curva,
        mode="lines", name=f"Curva MLP ({ativacao}/{solver})",
        line=dict(color="#7B1FA2", width=2, dash="solid"), opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=anos, y=valores,
        mode="lines+markers", name="Censos IBGE",
        line=dict(color="#1565C0", width=3),
        marker=dict(size=11, color="#0D47A1"),
    ))
    fig.add_trace(go.Scatter(
        x=ANOS_PROJECAO, y=vals_proj,
        mode="markers+text", name="Projeção",
        text=[f"{v:,.1f}" for v in vals_proj],
        textposition="top center",
        marker=dict(size=14, color="#E65100", symbol="star"),
    ))

    fig.update_layout(
        template="plotly_white", height=400,
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
    fig = px.bar(
        df, x="indicador_nome", y="valor", color="ano",
        barmode="group", text_auto=True,
        title=f"Visão geral — {municipio}",
        color_continuous_scale="Blues",
        labels={"indicador_nome": "Indicador", "valor": "Valor", "ano": "Ano"},
    )
    fig.update_layout(
        template="plotly_white", height=450,
        xaxis_tickangle=-35,
        margin=dict(l=40, r=20, t=60, b=120),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ESTILOS E COMPONENTES VISUAIS
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
          .badge-auto {
            display: inline-block;
            background: #2E7D32;
            color: white !important;
            padding: 0.12rem 0.5rem;
            border-radius: 999px;
            font-size: 0.72rem;
            margin-left: 0.4rem;
          }
          .badge-fixo {
            display: inline-block;
            background: #757575;
            color: white !important;
            padding: 0.12rem 0.5rem;
            border-radius: 999px;
            font-size: 0.72rem;
            margin-left: 0.4rem;
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


# ═══════════════════════════════════════════════════════════════════════════
# SEÇÕES DO PAINEL
# ═══════════════════════════════════════════════════════════════════════════

def render_municipio(cfg: dict) -> None:
    nome = cfg["nome"]
    df   = carregar_dados(cfg["pasta"], cfg["arquivo"])
    df_proj_precalc = carregar_projecoes(cfg["pasta"], cfg["arquivo"])

    st.markdown(
        f"""
        <div class="municipio-header">
          <span class="badge">Município</span>
          <h2 style="margin: 0.4rem 0 0.2rem; color:#0D47A1;">{nome}</h2>
          <p style="margin:0; color:#475569; font-size:0.95rem;">
            Dados do IBGE (Censos 1991–2022) com projeções MLP para
            {" e ".join(str(a) for a in ANOS_PROJECAO)}, usando otimização por indicador (LOOCV).
          </p>
        </div>
        """,
        unsafe_allow_html=True,
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
        return

    # ESTRUTURAÇÃO DA NAVEGAÇÃO EM SUB-ABAS POR MUNICÍPIO
    aba_geral, aba_detalhes, aba_tabela = st.tabs([
        "📊 Visão Geral & KPIs", 
        "📈 Análise por Indicador", 
        "🗂️ Tabela de Dados"
    ])

    # -------------------------------------------------------------------------
    # SUB-ABA 1: VISÃO GERAL & KPIS
    # -------------------------------------------------------------------------
    with aba_geral:
        n_indicadores = df["indicador_id"].nunique()
        anos_disp     = sorted(df["ano"].dropna().unique().astype(int))
        grupos        = sorted(df["grupo_censo"].unique()) if "grupo_censo" in df.columns else []
        n_auto        = df.drop_duplicates("indicador_id")["auto_selecionado"].sum() if "auto_selecionado" in df.columns else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Indicadores", n_indicadores)
        m2.metric("Período Censitário", f"{anos_disp[0]}–{anos_disp[-1]}" if anos_disp else "—")
        m3.metric("Grupos Temáticos", len(grupos))
        m4.metric("Modelos Otimizados", f"{int(n_auto)}/{n_indicadores}")

        if df_proj_precalc is None:
            st.caption(
                "ℹ️ CSV de projeções pré-calculadas não encontrado — "
                "as previsões estão sendo recalculadas em tempo real."
            )

        st.divider()
        st.plotly_chart(fig_barras_todos(df, nome), use_container_width=True)

    # -------------------------------------------------------------------------
    # SUB-ABA 2: ANÁLISE POR INDICADOR (COM FILTRO POR GRUPO TEMÁTICO)
    # -------------------------------------------------------------------------
    with aba_detalhes:
        col_grupo, col_ind = st.columns([1, 2])
        
        # Filtro 1: Grupo Temático
        if "grupo_censo" in df.columns:
            grupos_disponiveis = ["(Todos os Grupos)"] + sorted(df["grupo_censo"].dropna().unique().tolist())
            grupo_sel = col_grupo.selectbox("1. Filtrar por Grupo", grupos_disponiveis, key=f"grp_{nome}")
            df_filtrado = df if grupo_sel == "(Todos os Grupos)" else df[df["grupo_censo"] == grupo_sel]
        else:
            df_filtrado = df

        # Filtro 2: Indicador (atualizado dinamicamente)
        indicadores_disponiveis = sorted(df_filtrado["indicador_nome"].unique())
        ind_sel = col_ind.selectbox("2. Selecionar Indicador", indicadores_disponiveis, key=f"ind_{nome}")

        st.divider()

        # Detalhes do indicador selecionado
        sub = df[df["indicador_nome"] == ind_sel].sort_values("ano")
        anos     = sub["ano"].astype(int).tolist()
        vals     = sub["valor"].tolist()
        ylabel   = sub["unidade_medida"].iloc[0] if "unidade_medida" in sub.columns else "Valor"
        grupo    = sub["grupo_censo"].iloc[0] if "grupo_censo" in sub.columns else ""
        ind_id   = sub["indicador_id"].iloc[0]
        ativacao = sub["ativacao"].iloc[0] if "ativacao" in sub.columns else ATIVACAO_FALLBACK
        solver   = sub["solver"].iloc[0] if "solver" in sub.columns else SOLVER_FALLBACK
        auto     = bool(sub["auto_selecionado"].iloc[0]) if "auto_selecionado" in sub.columns else False
        loocv    = sub["loocv_mae"].iloc[0] if "loocv_mae" in sub.columns else np.nan

        if len(anos) < 2 or any(np.isnan(v) for v in vals):
            st.warning(f"**{ind_sel}** — série com menos de 2 pontos válidos; projeção indisponível.", icon="⚠️")
        else:
            badge_modelo = (
                f'<span class="badge-auto">auto — LOOCV MAE={loocv:,.2f}</span>'
                if auto and pd.notna(loocv)
                else f'<span class="badge-fixo">modelo fixo</span>'
            )
            st.markdown(f"**{ativacao} / {solver}** {badge_modelo}", unsafe_allow_html=True)

            vals_proj = _projecao_indicador(ind_id, anos, vals, ativacao, solver, df_proj_precalc)

            col_grafico, col_resumo = st.columns([3, 1])

            with col_grafico:
                st.plotly_chart(
                    fig_serie(ind_sel, anos, vals, ylabel, nome, ativacao, solver, vals_proj),
                    use_container_width=True,
                )

            with col_resumo:
                st.markdown("##### 📌 Projeções MLP")
                df_proj_tabela = pd.DataFrame({
                    "Ano": ANOS_PROJECAO,
                    f"Valor ({ylabel})": [round(v, 2) for v in vals_proj],
                })
                st.dataframe(df_proj_tabela, hide_index=True, use_container_width=True)
                st.caption(f"**Grupo:** {grupo}<br>**Censos na série:** {anos}", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SUB-ABA 3: DADOS BRUTOS
    # -------------------------------------------------------------------------
    with aba_tabela:
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


def render_comparativo(municipios_carregados: list[dict]) -> None:
    st.header("Comparativo entre municípios")
    st.caption(
        "Selecione um indicador para visualizar a evolução histórica "
        "e as projeções lado a lado."
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

    fig = go.Figure()
    cores = px.colors.qualitative.Bold
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
            mode="lines+markers", name=f"{mun} — Censos",
            line=dict(color=cor, width=3),
            marker=dict(size=10),
        ))

        vals_proj = _projecao_indicador(
            ind_id, anos, vals, ativacao, solver, dfs_proj.get(mun)
        )
        fig.add_trace(go.Scatter(
            x=ANOS_PROJECAO, y=vals_proj,
            mode="markers+text", name=f"{mun} — Projeção ({ativacao}/{solver})",
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
        
        if st.button("🔄 Recarregar Dados", help="Use para limpar o cache após subir novos dados no GitHub", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.caption("TCC — Projeções via MLP (sklearn), modelo por indicador escolhido via LOOCV")
        st.info(
            "Os dados têm como base o Censo do IBGE, o SIDRA e o Panorama "
            "do Censo. Parte das informações é estimada, podendo haver "
            "margem de erro para mais ou para menos.",
            icon="ℹ️",
        )
        st.divider()
        st.markdown(
            "**Instruções de Adição:**\n"
            "1. Rode o notebook para o município.\n"
            "2. Adicione os arquivos CSV na pasta `data/`.\n"
            "3. Configure a lista `MUNICIPIOS` no topo do código.\n"
        )

    nomes_abas = [cfg["nome"] for cfg in MUNICIPIOS] + ["🔀 Comparativo"]
    abas = st.tabs(nomes_abas)

    for aba, cfg in zip(abas[:-1], MUNICIPIOS):
        with aba:
            render_municipio(cfg)

    with abas[-1]:
        render_comparativo(MUNICIPIOS)


if __name__ == "__main__":
    main()
