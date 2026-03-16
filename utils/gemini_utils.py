import streamlit as st
import pandas as pd

_model = None

SYSTEM_PROMPT = """Você é um especialista em análise de dados censitários e políticas públicas
para o município de Castanhal – PA. Você tem acesso aos dados do Censo IBGE 2022 e
aos resultados de modelos de Machine Learning (classificação de vulnerabilidade
socioeconômica e regressão de infraestrutura urbana).

Diretrizes obrigatórias (Princípio de IA Ética — TCC Castanhal):
- Baseie suas respostas exclusivamente nos dados fornecidos no contexto.
- Cite números e indicadores específicos quando disponíveis.
- Indique claramente quando uma informação não está disponível nos dados.
- SEMPRE finalize respostas sobre grupos étnico-raciais, distribuição de renda ou
  fluxos migratórios com uma advertência sobre as limitações dos dados e possíveis
  vieses nas análises.
- SEMPRE encerre cada resposta com o aviso: "⚠️ Esta análise é baseada nos dados
  disponíveis do Censo 2022 e pode conter imprecisões. Use como apoio à decisão,
  não como diagnóstico definitivo."
- Nunca apresente resultados de ML como certeza absoluta — são estimativas estatísticas.
- Seja objetivo e use linguagem acessível a gestores públicos e pesquisadores.
"""

DISCLAIMER_IA = (
    "\n\n---\n⚠️ *Esta análise é baseada nos dados disponíveis do Censo IBGE 2022 "
    "e pode conter imprecisões. Use como apoio à decisão, não como diagnóstico definitivo.*"
)

TEMAS_DISPONIVEIS = [
    "Geral",
    "Demografia",
    "Domicílios",
    "Educação",
    "Trabalho & Renda",
    "Políticas Públicas",
]


def configurar_gemini() -> bool:
    """Inicializa o cliente Gemini. Retorna True se bem-sucedido."""
    global _model
    try:
        import google.generativeai as genai
        api_key = st.secrets["GEMINI_API_KEY"]
        if not api_key:
            raise ValueError("GEMINI_API_KEY está vazia.")
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT,
        )
        return True
    except (KeyError, FileNotFoundError):
        st.error(
            "❌ Chave da API Gemini não configurada. "
            "Adicione `GEMINI_API_KEY` nos Secrets do Streamlit Cloud."
        )
        return False
    except Exception as e:
        st.error(f"❌ Erro ao configurar Gemini: {e}")
        return False


def gerar_contexto_dados(df: pd.DataFrame, tema: str) -> str:
    """Gera resumo estatístico do DataFrame para injeção no prompt."""
    if df.empty:
        return f"Dados de {tema} não disponíveis no momento."
    stats = df.describe().round(2).to_string()
    n_setores = len(df)
    cols = ", ".join(df.columns.tolist()[:15])
    return (
        f"=== Dados do Censo 2022 de Castanhal — {tema} ===\n"
        f"Setores censitários analisados: {n_setores}\n"
        f"Variáveis disponíveis: {cols}\n\n"
        f"Estatísticas descritivas:\n{stats}"
    )


def gerar_contexto_tematico(tema: str, dataframes: dict) -> str:
    """Seleciona e prepara contexto baseado no tema escolhido pelo usuário."""
    mapa = {
        "Geral": list(dataframes.values()),
        "Demografia": [dataframes.get("demografico", pd.DataFrame())],
        "Domicílios": [dataframes.get("domicilios", pd.DataFrame())],
        "Educação": [dataframes.get("educacao", pd.DataFrame())],
        "Trabalho & Renda": [dataframes.get("trabalho_renda", pd.DataFrame())],
        "Políticas Públicas": [
            dataframes.get("features_compostas", pd.DataFrame())
        ],
    }
    dfs = mapa.get(tema, [])
    partes = []
    for df in dfs:
        if df is not None and not df.empty:
            partes.append(gerar_contexto_dados(df, tema))
    return "\n\n".join(partes) if partes else f"Dados de {tema} não disponíveis."


def consultar_ia(pergunta: str, contexto: str, historico: list[dict]) -> str:
    """Consulta o Gemini com pergunta, contexto de dados e histórico da conversa."""
    global _model
    if _model is None:
        if not configurar_gemini():
            return "⚠️ Assistente de IA não disponível. Verifique a configuração da API key."
    try:
        hist_texto = ""
        for msg in historico[-6:]:  # últimas 3 trocas
            role = "Usuário" if msg["role"] == "user" else "Assistente"
            hist_texto += f"{role}: {msg['content']}\n"

        prompt = (
            f"Contexto dos dados:\n{contexto}\n\n"
            f"Histórico recente:\n{hist_texto}\n"
            f"Nova pergunta: {pergunta}"
        )
        response = _model.generate_content(prompt)
        return response.text + DISCLAIMER_IA
    except Exception as e:
        return f"⚠️ Erro ao consultar a IA: {e}. Tente novamente em instantes."
