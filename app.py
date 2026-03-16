import streamlit as st

st.set_page_config(
    page_title="Sistema de Inteligência Territorial de Castanhal",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Flag_of_Brazil.svg/320px-Flag_of_Brazil.svg.png",
        width=60,
    )
    st.markdown("## 🏙️ Castanhal – PA")
    st.markdown("**Sistema de Inteligência Territorial**")
    st.divider()
    st.markdown(
        """
        Plataforma de análise dos dados do **Censo 2022** do IBGE para o município
        de Castanhal, com modelos de Machine Learning e assistente de IA.
        """
    )
    st.divider()
    st.caption("Projeto TCC • Dados: IBGE Censo 2022")
    st.caption("Pipeline: Drive → Colab → GitHub → Streamlit")

st.title("🏙️ Sistema de Inteligência Territorial de Castanhal")
st.markdown(
    """
    Bem-vindo! Use o **menu lateral** para navegar entre as análises:

    | Página | Conteúdo |
    |--------|----------|
    | 📊 **Dinâmica Populacional** | Pirâmide etária, distribuição étnico-racial, densidade |
    | 🏠 **Diagnóstico Habitacional** | Condições de moradia, saneamento, IAH |
    | 📚 **Educação & Renda** | Escolaridade, trabalho, distribuição de renda |
    | 🤖 **Machine Learning** | Resultados dos modelos IVS e IAH |
    | 🏛️ **Políticas Públicas** | Recomendações baseadas nos dados e modelos |
    | 💬 **Assistente IA** | Perguntas em linguagem natural (Gemini) |
    """
)
st.divider()
st.info(
    "ℹ️ Os dados são carregados diretamente do repositório GitHub. "
    "Execute o notebook Colab para atualizar os artefatos.",
    icon="ℹ️",
)
