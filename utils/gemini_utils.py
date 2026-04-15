import streamlit as st
import pandas as pd

from utils.censo_projecoes import SERIES, dataframe_relatorio_completo, projetar_mlp

_model = None

SYSTEM_PROMPT_MODO_DADOS = """Você é um assistente acadêmico especializado nos dados censitários de Castanhal – PA
extraídos do trabalho de conclusão de curso (notebook Arquivo_tcc.ipynb) e nas projeções com redes neurais (sklearn MLP).

Regras:
- Priorize o CONTEXTO DOS DADOS anexado à mensagem; não invente números que não estejam lá.
- Explique de forma clara o que os indicadores revelam, limitações (poucos pontos temporais, extrapolação) e possíveis usos para políticas públicas.
- Para grupos étnico-raciais, renda e populações indígenas, mencione limitações e possíveis vieses.
- Deixe claro que previsões são modelos estatísticos, não projeções oficiais do IBGE.
- Seja útil sobre "o que pode ser feito para melhorar indicadores" com base em evidências gerais e nos dados fornecidos.
- Responda em português do Brasil.
"""

SYSTEM_PROMPT_MODO_LIVRE = """Você é um assistente de Engenharia de Dados e Ciência de Dados.
O usuário pode perguntar qualquer tema (ferramentas, arquitetura, boas práticas, carreira, SQL, cloud, etc.).
Responda em português do Brasil, de forma didática e objetiva.
Se a pergunta não for relacionada a dados, responda normalmente sem forçar o contexto do Censo.
Não afirme ter acesso a dados internos do município neste modo, salvo se o usuário colar na conversa.
"""

DISCLAIMER_RESPOSTA = (
    "\n\n---\n⚠️ *Esta resposta é informativa e pode conter imprecisões. "
    "No modo Dados do Censo, use como apoio à análise — não como decisão única.*"
)


def texto_contexto_notebook_completo() -> str:
    """Texto denso com todas as séries e previsões MLP para injeção no Gemini (modo Dados)."""
    partes = [
        "=== CASTANHAL-PA — SÉRIES DO NOTEBOOK (Arquivo_tcc.ipynb) ===",
        "Fonte: tabelas IBGE comparativos (1991, 2000, 2010, 2022) conforme usadas no TCC.",
        "",
    ]
    for s in SERIES:
        pr = projetar_mlp(s.anos, s.valores)
        prev_txt = ", ".join(f"{a}: {v:.4f}" for a, v in pr["previsoes"])
        hist = ", ".join(f"{ano}: {val}" for ano, val in zip(s.anos, s.valores))
        partes.append(f"• [{s.area}] {s.titulo} ({s.unidade})")
        partes.append(f"  Histórico: {hist}")
        partes.append(f"  Previsões MLP (ilustrativas): {prev_txt}")
        partes.append(f"  Nota: {s.nota_prev}")
        partes.append("")
    try:
        df = dataframe_relatorio_completo()
        partes.append("=== TABELA RESUMO (colunas) ===")
        partes.append(df.head(30).to_string())
    except Exception:
        pass
    return "\n".join(partes)


def configurar_gemini(system_instruction: str | None = None) -> bool:
    """Inicializa o cliente Gemini. system_instruction opcional sobrescreve o padrão do modo dados."""
    global _model
    try:
        import google.generativeai as genai

        api_key = st.secrets["GEMINI_API_KEY"]
        if not api_key:
            raise ValueError("GEMINI_API_KEY está vazia.")
        genai.configure(api_key=api_key)
        instr = system_instruction or SYSTEM_PROMPT_MODO_DADOS
        _model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=instr,
        )
        return True
    except (KeyError, FileNotFoundError):
        st.error(
            "Chave da API Gemini não configurada. Adicione `GEMINI_API_KEY` em `.streamlit/secrets.toml` "
            "ou nos Secrets do Streamlit Cloud."
        )
        return False
    except Exception as e:
        st.error(f"Erro ao configurar Gemini: {e}")
        return False


def reset_model():
    """Força recriação do modelo ao mudar modo (instrução de sistema diferente)."""
    global _model
    _model = None


def consultar_gemini_modo(
    pergunta: str,
    historico: list[dict],
    modo: str,
    contexto_dados_censo: str | None = None,
) -> str:
    """
    modo: 'dados' — usa contexto do Censo + SYSTEM_PROMPT_MODO_DADOS
          'livre' — pergunta aberta, SYSTEM_PROMPT_MODO_LIVRE
    """
    system = SYSTEM_PROMPT_MODO_DADOS if modo == "dados" else SYSTEM_PROMPT_MODO_LIVRE
    if not configurar_gemini(system_instruction=system):
        return "Configure GEMINI_API_KEY nos secrets para usar o chat."

    global _model
    try:
        hist_texto = ""
        for msg in historico[-8:]:
            role = "Usuário" if msg["role"] == "user" else "Assistente"
            hist_texto += f"{role}: {msg['content']}\n"

        if modo == "dados":
            ctx = contexto_dados_censo or texto_contexto_notebook_completo()
            bloco = f"Contexto (Censo Castanhal / notebook):\n{ctx}\n\n"
        else:
            bloco = ""

        prompt = (
            f"{bloco}"
            f"Histórico recente:\n{hist_texto}\n"
            f"Pergunta atual: {pergunta}"
        )
        response = _model.generate_content(prompt)
        return (response.text or "") + DISCLAIMER_RESPOSTA
    except Exception as e:
        return f"Erro ao consultar o Gemini: {e}"


# Compatibilidade com imports antigos
TEMAS_DISPONIVEIS = ["Geral", "Modo dados do Censo", "Modo livre"]

DISCLAIMER_RESPOSTA_LEGACY = DISCLAIMER_RESPOSTA


def gerar_contexto_dados(df: pd.DataFrame, tema: str) -> str:
    if df.empty:
        return f"Dados de {tema} não disponíveis no momento."
    stats = df.describe().round(2).to_string()
    n = len(df)
    cols = ", ".join(df.columns.tolist()[:20])
    return (
        f"=== {tema} ===\n"
        f"Linhas: {n}\nVariáveis: {cols}\n\nEstatísticas:\n{stats}"
    )


def consultar_gemini(pergunta: str, contexto: str, historico: list[dict]) -> str:
    """API legada: trata como modo dados com contexto customizado."""
    return consultar_gemini_modo(
        pergunta,
        historico,
        modo="dados",
        contexto_dados_censo=contexto,
    )
