"""
Painel Streamlit — TCC Castanhal-PA (Censo / notebook Arquivo_tcc.ipynb).
Multipágina via st.navigation; tema em .streamlit/config.toml.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from utils.censo_projecoes import (
    SERIES,
    projetar_mlp,
    series_com_pelo_menos_2_censos_no_periodo,
    dataframe_relatorio_completo,
)
from utils.gemini_utils import (
    consultar_gemini_modo,
    texto_contexto_notebook_completo,
)
from utils.relatorio_export import csv_bytes, pdf_bytes
from utils.a11y import (
    TEXTO_DADOS,
    TEXTO_DOWNLOAD,
    TEXTO_HOME,
    TEXTO_INFO,
    TEXTO_PERGUNTAS,
    TEXTO_TRILHA,
    bloco_texto_leitores_ecra,
    render_ouvir_descricao,
)
from utils.trilha_censo import TRILHA_PASSOS, serie_por_id, texto_tts_passo
from utils.castanhal_apresentacao import markdown_apresentacao

# ── Página ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Castanhal em dados | TCC",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊",
)


@st.cache_data(ttl=3600, show_spinner=False)
def _contexto_gemini_cache() -> str:
    return texto_contexto_notebook_completo()


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.2rem; max-width: 1200px; }
          div[data-testid="stSidebarNav"] { font-weight: 500; }
          .tcc-hero {
            background: linear-gradient(135deg, #E3F2FD 0%, #FFFFFF 55%, #FFF8E1 100%);
            padding: 1.75rem 1.5rem;
            border-radius: 18px;
            border: 1px solid #BBDEFB;
            margin-bottom: 1.25rem;
          }
          .tcc-badge {
            display: inline-block;
            background: #1565C0;
            color: white !important;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            letter-spacing: 0.03em;
          }
          .stTabs [data-baseweb="tab-list"] { gap: 8px; }
          .a11y-hint { font-size: 0.85rem; color: #475569; margin-bottom: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _fig_indicador_serie(s) -> go.Figure:
    pr = projetar_mlp(s.anos, s.valores)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(s.anos),
            y=list(s.valores),
            mode="lines+markers",
            name="Censos IBGE",
            line=dict(color="#1565C0", width=3),
            marker=dict(size=11, color="#0D47A1"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pr["anos_curva"],
            y=pr["val_curva"],
            mode="lines",
            name="Curva MLP (treinada)",
            line=dict(color="#7B1FA2", width=2, dash="solid"),
            opacity=0.85,
        )
    )
    for af, pv in pr["previsoes"]:
        fig.add_trace(
            go.Scatter(
                x=[af],
                y=[pv],
                mode="markers+text",
                name=f"Projeção {af}",
                text=[f"{pv:.2f}"],
                textposition="top center",
                marker=dict(size=14, color="#E65100", symbol="star", line=dict(width=1, color="#FFF")),
            )
        )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        title=dict(text=f"<b>{s.titulo}</b> <span style='font-size:12px'>({s.unidade})</span>", x=0.02),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(title="Ano")
    fig.update_yaxes(title=s.unidade)
    return fig


def _render_a11y_bloco(texto: str, key: str) -> None:
    """Descrição textual + botão TTS (voz do navegador)."""
    st.caption("♿ Acessibilidade: descrição abaixo; use «Ouvir descrição» para áudio opcional.")
    st.markdown(bloco_texto_leitores_ecra(texto))
    render_ouvir_descricao(texto, key=key)


def render_home() -> None:
    _render_a11y_bloco(TEXTO_HOME, "tts_home")
    st.markdown(
        """
        <div class="tcc-hero">
          <span class="tcc-badge">Trabalho de Conclusão de Curso</span>
          <h1 style="margin:0.6rem 0 0.4rem 0; color:#0D47A1;">Castanhal em perspectiva censitária</h1>
          <p style="font-size:1.05rem; line-height:1.55; color:#334155; margin:0;">
            Este painel apoia a leitura dos <strong>dados agregados do IBGE</strong> e das
            <strong>projeções com redes neurais</strong> construídas no notebook
            <code>Arquivo_tcc.ipynb</code> — um estudo que combina Ciência de Dados e visualização
            para o município de <strong>Castanhal – PA</strong>.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown(markdown_apresentacao())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("População (2022)", "192.256", "habitantes", help="Total residente — série comparativa do notebook.")
    with c2:
        st.metric("Renda per capita (2022)", "R$ 1.055", "nominal", help="Rendimento domiciliar mensal per capita.")
    with c3:
        st.metric("Indicadores modelados", str(len(SERIES)), "séries", help="Quantidade de séries com gráficos MLP no TCC.")
    with c4:
        st.metric("Censos na base", "1991–2022", "recortes", help="Conforme disponibilidade por indicador.")

    st.subheader("Panorama dinâmico")
    p1, p2 = st.columns(2, gap="large")

    df_pop = pd.DataFrame(
        {
            "Censo": ["1991", "2000", "2010", "2022"],
            "População": [102_071, 134_496, 173_149, 192_256],
        }
    )
    fig_bar = px.bar(
        df_pop,
        x="Censo",
        y="População",
        color="Censo",
        color_discrete_sequence=px.colors.qualitative.Bold,
        text_auto=",",
        title="População total — evolução entre censos",
    )
    fig_bar.update_layout(template="plotly_white", height=360, showlegend=False)
    p1.plotly_chart(fig_bar, use_container_width=True)

    # Sunburst composição cor/raça 2022 (valores absolutos do notebook)
    labels = ["Castanhal 2022", "Parda", "Branca", "Preta", "Amarela", "Indígena"]
    parents = ["", "Castanhal 2022", "Castanhal 2022", "Castanhal 2022", "Castanhal 2022", "Castanhal 2022"]
    values = [192_256, 132_079, 42_881, 16_429, 719, 144]
    fig_sb = px.sunburst(
        names=labels,
        parents=parents,
        values=values,
        color=labels,
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Composição por cor ou raça (hab.) — referência 2022",
    )
    fig_sb.update_layout(height=360, margin=dict(t=50, l=10, r=10, b=10))
    p2.plotly_chart(fig_sb, use_container_width=True)

    df_renda = pd.DataFrame(
        {
            "Ano": [1991, 2000, 2010, 2022],
            "R$ per capita": [160.54, 260.98, 467.32, 1055.05],
        }
    )
    fig_line = px.area(
        df_renda,
        x="Ano",
        y="R$ per capita",
        title="Rendimento domiciliar mensal per capita (R$)",
        color_discrete_sequence=["#00897B"],
    )
    fig_line.update_traces(fill="tozeroy", line=dict(width=3))
    fig_line.update_layout(template="plotly_white", height=340)
    st.plotly_chart(fig_line, use_container_width=True)

    st.caption(
        "Valores conforme extraídos do notebook de análise. Visualizações com Plotly — "
        "passe o cursor para detalhes."
    )


def render_trilha() -> None:
    """
    Trilha tipo «dungeon» Streamlit: passos em session_state, narrativa + gráfico por sala.
    Inspirado em apps interativos por passos (ex.: dungeon.streamlit.app).
    """
    total = len(TRILHA_PASSOS)
    if "trilha_passo" not in st.session_state:
        st.session_state.trilha_passo = 0
    i = int(st.session_state.trilha_passo)
    i = max(0, min(i, total - 1))
    st.session_state.trilha_passo = i

    passo = TRILHA_PASSOS[i]
    tts = TEXTO_TRILHA + " " + texto_tts_passo(passo, i, total)
    _render_a11y_bloco(tts, f"tts_trilha_{i}_{passo['id']}")

    st.title("Trilha interativa — Castanhal em dados")
    st.caption(
        "Avance por «salas» temáticas. Cada etapa usa **dados reais** do notebook do TCC — "
        "estilo jogo por passos, semelhante a templates interativos no Streamlit Community Cloud."
    )
    st.caption(f"Progresso: sala {i + 1} de {total}")
    st.progress((i + 1) / total)

    st.markdown(f"### {passo['titulo']}")
    st.markdown(passo["narrativa"])

    sid = passo.get("serie_id")
    if sid:
        s = serie_por_id(sid)
        if s is not None:
            st.plotly_chart(_fig_indicador_serie(s), use_container_width=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("◀ Voltar", disabled=i <= 0, key="trilha_voltar"):
            st.session_state.trilha_passo = i - 1
            st.rerun()
    with b2:
        if st.button("Reiniciar trilha", key="trilha_reset"):
            st.session_state.trilha_passo = 0
            st.rerun()
    with b3:
        if st.button("Avançar ▶", disabled=i >= total - 1, key="trilha_avanc"):
            st.session_state.trilha_passo = i + 1
            st.rerun()

    if i >= total - 1:
        st.success("Trilha concluída! Explore **Dados** para o painel completo ou **Download** para os ficheiros.")


def render_dados() -> None:
    _render_a11y_bloco(TEXTO_DADOS, "tts_dados")
    st.title("Dados e projeções (MLP)")
    st.markdown(
        "Cada bloco reproduz a **lógica do notebook**: escalonamento, `MLPRegressor` "
        "(duas camadas de 10 neurônios, solver `lbfgs`) e pontos futuros ilustrativos."
    )

    st.subheader("Indicadores com histórico em ao menos dois censos (1991, 2000, 2010 ou 2022)")
    sub = series_com_pelo_menos_2_censos_no_periodo()
    rows = []
    for s in sub:
        pr = projetar_mlp(s.anos, s.valores)
        pv = ", ".join(f"{a}: {v:.2f}" for a, v in pr["previsoes"])
        rows.append(
            {
                "Área": s.area,
                "Indicador": s.titulo,
                "Censos no modelo": ", ".join(str(x) for x in s.anos),
                "Nº censos": len(s.anos),
                "Projeções (MLP)": pv,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.info(
        "Somente indicadores cuja série cruza **pelo menos dois** desses anos censitários entram neste recorte.",
        icon="📌",
    )

    areas = sorted({s.area for s in SERIES})
    tabs = st.tabs(areas)
    for tab, area in zip(tabs, areas):
        with tab:
            series_area = [s for s in SERIES if s.area == area]
            for s in series_area:
                st.markdown(f"#### {s.titulo}")
                st.plotly_chart(_fig_indicador_serie(s), use_container_width=True)
                st.markdown(
                    f"**O que os dados mostram:** {s.descricao_curta}\n\n"
                    f"**Metodologia (notebook):** {s.o_que_foi_feito}\n\n"
                    f"**Projeção:** {s.nota_prev}"
                )
                st.divider()


EXEMPLOS_PERGUNTAS = [
    "Quais indicadores têm maior incerteza nas projeções e por quê?",
    "Como o rendimento per capita evoluiu e o que as projeções sugerem para políticas de renda?",
    "O que explica o perfil de população urbana versus rural em Castanhal?",
    "Quais ações poderiam melhorar o componente educação do IDH, com base na série histórica?",
    "Como interpretar com responsabilidade os dados de cor ou raça entre 2010 e 2022?",
]


def render_perguntas() -> None:
    _render_a11y_bloco(TEXTO_PERGUNTAS, "tts_perg")
    st.title("Perguntas ao assistente (Gemini)")
    st.caption("Configure `GEMINI_API_KEY` em `.streamlit/secrets.toml` (ou Secrets no Cloud).")

    modo = st.radio(
        "Modo de conversa",
        options=["dados", "livre"],
        format_func=lambda x: "📊 Dados do Censo (notebook + projeções)"
        if x == "dados"
        else "🌐 Livre (engenharia de dados e temas gerais)",
        horizontal=True,
    )

    if modo == "dados":
        st.info(
            "O modelo usa o **texto consolidado** das séries e previsões MLP do notebook "
            "`Arquivo_tcc.ipynb` como contexto.",
            icon="ℹ️",
        )
    else:
        st.info("Modo livre: sem injeção automática das tabelas do Censo.", icon="🌐")

    st.markdown("**Sugestões de pergunta:**")
    st.caption(" • ".join(EXEMPLOS_PERGUNTAS[:3]) + " …")
    if st.button("Limpar histórico do chat", type="secondary"):
        st.session_state["chat_msgs"] = []
        st.rerun()

    if "chat_msgs" not in st.session_state:
        st.session_state["chat_msgs"] = []

    for msg in st.session_state["chat_msgs"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Escreva sua pergunta…")

    if prompt:
        st.session_state["chat_msgs"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        ctx = _contexto_gemini_cache() if modo == "dados" else None
        with st.chat_message("assistant"):
            with st.spinner("Gerando resposta…"):
                resposta = consultar_gemini_modo(
                    prompt,
                    st.session_state["chat_msgs"][:-1],
                    modo=modo,
                    contexto_dados_censo=ctx,
                )
            st.markdown(resposta)
        st.session_state["chat_msgs"].append({"role": "assistant", "content": resposta})


def render_info() -> None:
    _render_a11y_bloco(TEXTO_INFO, "tts_info")
    st.title("Sobre este TCC")
    st.info(
        "**Acessibilidade:** cada página inclui uma **descrição em texto** para leitores de ecrã e um botão "
        "**«Ouvir descrição»**, que usa a **síntese de voz do próprio navegador** (Web Speech API, português quando disponível). "
        "O áudio é opcional e complementar — não substitui ferramentas de acessibilidade do sistema.",
        icon="♿",
    )
    st.markdown(markdown_apresentacao())
    st.markdown(
        "### Trabalho de Conclusão de Curso\n\n"
        "**Autor:** Luan Evaristo de Melo Lindolfo  \n"
        "**Tema:** Ciência de dados aplicada a indicadores censitários de Castanhal–PA, "
        "com projeções via redes neurais e divulgação em ambiente web (Streamlit).\n\n"
        "**Funcionamento deste painel:**\n"
        "- **Home:** síntese contextual e infográficos sobre a população e a renda.  \n"
        "- **Trilha:** jogo interativo por «salas» — avance e desbloqueie gráficos reais dos censos.  \n"
        "- **Dados:** séries extraídas do notebook, alinhadas aos gráficos de rede neural, com texto explicativo.  \n"
        "- **Perguntas:** chat com Gemini — modo **dados** (com contexto do notebook) ou **livre**.  \n"
        "- **Download:** exportação CSV/PDF com tabela de indicadores e projeções.  \n\n"
        "**Pipeline conceitual (mesma ideia do estudo):**"
    )

    diagram = """
flowchart LR
  A[Colab + Drive] --> B[CSV IBGE comparativos]
  B --> C[DataFrames por tema]
  C --> D[Agrupar indicadores por nº de censos]
  C --> E[Modelos: RNA poly SVR ARIMA ensemble...]
  D --> E
  E --> F[Melhores gráficos / painéis]
  F --> G[Push para GitHub / uso Streamlit]
"""
    components.html(
        f"""
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
</head>
<body style="margin:0;background:#F8FAFC;">
  <script>mermaid.initialize({{startOnLoad:true, theme: 'neutral'}});</script>
  <div class="mermaid">
{diagram}
  </div>
</body>
</html>
""",
        height=280,
    )
    with st.expander("Ver diagrama em texto (Mermaid)"):
        st.code(diagram.strip(), language="text")

    st.markdown(
        "---\n\n### Agradecimento sugerido\n\n"
        "> *Obrigado por acompanhar este estudo!*  \n"
        "> *Este projeto une Ciência de Dados e Desenvolvimento Web para entender o passado "
        "e projetar o futuro de Castanhal–PA.*  \n"
        "> *Desenvolvido com dedicação por **Luan Evaristo de Melo Lindolfo**.*"
    )


def render_download() -> None:
    _render_a11y_bloco(TEXTO_DOWNLOAD, "tts_dl")
    st.title("Download de relatórios")
    st.markdown(
        "Baixe a **tabela consolidada** (indicadores históricos e colunas `previsto_*` das projeções MLP) "
        "para reutilização em planilhas ou apresentações. O PDF é um resumo textual automático."
    )

    df_preview = dataframe_relatorio_completo()
    st.dataframe(df_preview.head(25), use_container_width=True)

    b1, b2 = st.columns(2)
    with b1:
        st.download_button(
            label="Baixar CSV (UTF-8)",
            data=csv_bytes(),
            file_name="castanhal_indicadores_projecoes_mlp.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )
    with b2:
        try:
            pdf = pdf_bytes()
            st.download_button(
                label="Baixar PDF (resumo)",
                data=pdf,
                file_name="castanhal_relatorio_tcc.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except ImportError:
            st.warning("Instale `fpdf2` para habilitar o PDF: `pip install fpdf2`")


# ── Navegação principal ───────────────────────────────────────────────────────
_inject_css()

pages = [
    st.Page(render_home, title="Home", icon="🏠", default=True),
    st.Page(render_trilha, title="Trilha", icon="🎮"),
    st.Page(render_dados, title="Dados", icon="📈"),
    st.Page(render_perguntas, title="Perguntas", icon="💬"),
    st.Page(render_info, title="Info", icon="ℹ️"),
    st.Page(render_download, title="Download", icon="⬇️"),
]

nav = st.navigation(pages)
with st.sidebar:
    st.markdown("### Castanhal · dados")
    st.caption("TCC — notebook `Arquivo_tcc.ipynb` + Streamlit")
    st.divider()
nav.run()
