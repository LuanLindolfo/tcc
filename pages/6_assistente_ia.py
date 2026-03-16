import streamlit as st

from utils.gemini_utils import (
    configurar_gemini,
    gerar_contexto_tematico,
    consultar_ia,
    TEMAS_DISPONIVEIS,
)
from utils.data_loader import (
    load_demografico,
    load_domicilios,
    load_educacao,
    load_trabalho_renda,
    load_features_compostas,
)

st.set_page_config(page_title="Assistente IA — Castanhal", layout="wide")

st.title("💬 Assistente de IA — Gemini")
st.markdown(
    "Faça perguntas em linguagem natural sobre os dados do Censo 2022 de Castanhal. "
    "As respostas são geradas por IA e devem ser tratadas como **apoio à análise**, "
    "não como diagnóstico definitivo."
)
st.divider()

# ── Inicializar estado da conversa ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "👋 Olá! Sou o assistente de IA do Sistema de Inteligência Territorial de Castanhal.\n\n"
                "Posso ajudar com perguntas sobre os dados do Censo 2022. Exemplos:\n"
                "- *Qual é a taxa de analfabetismo em Castanhal?*\n"
                "- *Quais setores têm maior vulnerabilidade socioeconômica?*\n"
                "- *Como a renda se distribui entre os bairros?*\n"
                "- *Quais políticas públicas são recomendadas para habitação?*\n\n"
                "Selecione um tema no menu ao lado para que eu tenha mais contexto sobre os dados! 📊"
            ),
        }
    ]

# ── Sidebar: seletor de tema ──────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Configurações")
    tema = st.selectbox("📂 Tema dos dados:", TEMAS_DISPONIVEIS)
    st.caption("O tema selecionado injeta estatísticas dos dados correspondentes no contexto da IA.")

    if st.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ── Carregar dados para contexto ──────────────────────────────────────────────
@st.cache_data(ttl=3600)
def _carregar_todos():
    return {
        "demografico": load_demografico(),
        "domicilios": load_domicilios(),
        "educacao": load_educacao(),
        "trabalho_renda": load_trabalho_renda(),
        "features_compostas": load_features_compostas(),
    }

with st.spinner("Preparando contexto dos dados..."):
    dataframes = _carregar_todos()

# ── Exibir histórico da conversa ──────────────────────────────────────────────
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input da pergunta ─────────────────────────────────────────────────────────
pergunta = st.chat_input("Digite sua pergunta sobre os dados de Castanhal...")

if pergunta:
    st.session_state["messages"].append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Consultando IA..."):
            contexto = gerar_contexto_tematico(tema, dataframes)
            resposta = consultar_ia(
                pergunta,
                contexto,
                st.session_state["messages"][:-1],
            )
        st.markdown(resposta)

    st.session_state["messages"].append({"role": "assistant", "content": resposta})
