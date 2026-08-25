# -*- coding: utf-8 -*-
"""
app.py — Painel Streamlit TCC (coringa multi-município)
Edite apenas o bloco CONFIGURAÇÃO para adicionar novas cidades.

Ajustado para o notebook v2 (seleção automática de ativação/solver por
LOOCV): o app NÃO usa mais 'relu'/'lbfgs' fixo. Ele lê, por indicador,
qual combinação foi escolhida pelo notebook (colunas 'ativacao'/'solver'
no CSV histórico) e, quando disponível, usa diretamente as previsões
2030/2040 já calculadas pelo notebook (CSV de projeções) — em vez de
retreinar com um modelo genérico que poderia divergir do notebook.
Se o CSV de projeções não existir (notebook antigo/ainda não gerado),
o app recalcula em tempo real usando a ativação/solver salvos por
indicador (ou 'relu'/'lbfgs' como último fallback, para CSVs antigos
que nem tinham essas colunas).
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
# ║              BLOCO DE CONFIGURAÇÃO — EDITE APENAS AQUI                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Título geral do painel
TITULO_PAINEL = "Censo IBGE — Projeções Municipais"

# Cada entrada é um município.
# "pasta" aponta para a pasta de dados dentro do repositório.
# "arquivo" é o nome do CSV histórico exportado pelo notebook (seção 7).
# "nome"    é o rótulo exibido na aba.
# O CSV de projeções (2030/2040) é localizado automaticamente a partir
# do nome do "arquivo" — não precisa declarar mais nada aqui.
MUNICIPIOS = [
    {
        "nome":    "Belém",
        "pasta":   "data_belem",
        "arquivo": "indicadores_belem_tratados.csv",
    },
    # Para adicionar uma nova cidade:
    # {
    #     "nome":    "Nome da Cidade",
    #     "pasta":   "data_nomecidade",
    #     "arquivo": "indicadores_nomecidade_tratados.csv",
    # },
]

# Anos futuros para projeção (usados só quando o app precisa recalcular
# porque o CSV de projeções do notebook ainda não existe)
ANOS_PROJECAO = [2030, 2040]

# Parâmetros do modelo MLP (mesmos do notebook) — usados apenas como
# fallback de treino quando o notebook não informou ativação/solver
ATIVACAO_FALLBACK = "relu"
SOLVER_FALLBACK   = "lbfgs"
HIDDEN_LAYERS     = (10, 10)
RANDOM_STATE      = 42
MAX_ITER          = 5000

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     FIM DO BLOCO DE CONFIGURAÇÃO                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ═══════════════════════════════════════════════════════════════════════════

def _raiz_repo() -> str:
    """Retorna o diretório raiz do repositório (onde app.py está)."""
    return os.path.dirname(os.path.abspath(__file__))


def _arquivo_projecoes(arquivo_historico: str) -> str:
    """
    Deriva o nome do CSV de projeções a partir do nome do CSV histórico,
    seguindo o padrão do notebook:
      indicadores_<municipio>_tratados.csv → projecoes_<municipio>_2030_2040.csv
    Não exige configuração extra em MUNICIPIOS.
    """
    nome = arquivo_historico
    if nome.startswith("indicadores_") and nome.endswith("_tratados.csv"):
        meio = nome[len("indicadores_"):-len("_tratados.csv")]
        anos_tag = "_".join(str(a) for a in ANOS_PROJECAO)
        return f"projecoes_{meio}_{anos_tag}.csv"
    # nome fora do padrão esperado — sem CSV de projeções correspondente
    return ""


@st.cache_data(show_spinner=False)
def carregar_dados(pasta: str, arquivo: str) -> pd.DataFrame | None:
    """Lê o CSV histórico de indicadores gerado pelo notebook. None se não existir."""
    caminho = os.path.join(_raiz_repo(), pasta, arquivo)
    if not os.path.exists(caminho):
        return None
    df = pd.read_csv(caminho)
    if "valor" in df.columns:
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    # Compatibilidade com CSVs antigos, gerados antes da seleção automática
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


@st.cache_data(show_spinner=False)
def carregar_projecoes(pasta: str, arquivo_historico: str) -> pd.DataFrame | None:
    """
    Lê o CSV de projeções 2030/2040 já calculado pelo notebook (mesmo
    modelo escolhido por LOOCV). Retorna None se ainda não existir —
    nesse caso o app recalcula sob demanda.
    """
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


def _treinar_mlp(anos: list[int], valores: list[float], anos_alvo: list[int],
                 ativacao: str = ATIVACAO_FALLBACK,
                 solver: str = SOLVER_FALLBACK) -> list[float]:
    """
    Treina um MLP com a ativação/solver informados (por padrão, o que o
    notebook escolheu para aquele indicador específico via LOOCV — não
    um valor fixo) e retorna previsões para anos_alvo.
    """
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


def _curva_mlp(anos: list[int], valores: list[float],
               ativacao: str, solver: str) -> tuple[list[int], list[float]]:
    """Gera curva contínua para o gráfico, usando a mesma ativação/solver do ponto previsto."""
    anos_curva = list(range(min(anos) - 2, 2046))
    vals_curva = _treinar_mlp(anos, valores, anos_curva, ativacao, solver)
    return anos_curva, vals_curva


def _projecao_indicador(ind_id: str, anos: list[int], valores: list[float],
                        ativacao: str, solver: str,
                        df_proj_precalc: pd.DataFrame | None) -> list[float]:
    """
    Retorna a projeção 2030/2040 para o indicador. Prioriza o valor já
    calculado pelo notebook (df_proj_precalc, casado por indicador_id);
    se não houver, recalcula em tempo real com a mesma ativação/solver.
    """
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
    """Gráfico de linha com dados reais + curva MLP + projeções já calculadas."""
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


def render_municipio(cfg: dict) -> None:
    """Renderiza a aba completa de um município."""
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
            {" e ".join(str(a) for a in ANOS_PROJECAO)}, usando a
            ativação/solver escolhida automaticamente por indicador
            (validação leave-one-out) no notebook.
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
        "e as projeções lado a lado. Cada município usa a ativação/solver "
        "escolhida individualmente pelo seu próprio notebook."
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
        st.caption("TCC — Projeções via MLP (sklearn), modelo por indicador escolhido via LOOCV")
        st.divider()
        st.markdown(
            "**Como adicionar uma cidade:**\n"
            "1. Rode o notebook coringa para o município.\n"
            "2. Faça push da pasta `data_<cidade>/` com os CSVs "
            "(`indicadores_..._tratados.csv` e, se já gerado, "
            "`projecoes_..._2030_2040.csv`).\n"
            "3. Adicione a entrada em `MUNICIPIOS` no topo de `app.py` "
            "(apenas nome, pasta e nome do CSV histórico — o CSV de "
            "projeções é localizado automaticamente).\n"
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
