"""
Acessibilidade: descrições em texto + botão «Ouvir descrição» (Web Speech API, pt-BR).
O áudio usa a voz do navegador; não substitui leitor de ecrã nem gravação profissional.
"""

from __future__ import annotations

import json

import streamlit.components.v1 as components


def render_ouvir_descricao(
    texto: str,
    *,
    label: str = "Ouvir descrição desta página",
    key: str = "tts_default",
) -> None:
    """Lê `texto` em voz alta no browser do utilizador (após clicar)."""
    safe_js = json.dumps(texto, ensure_ascii=False)
    uid = "".join(c if c.isalnum() else "_" for c in key)[:64]
    html = f"""
<div style="font-family:system-ui,sans-serif;margin:0 0 12px 0;">
  <button type="button" id="btn_speak_{uid}"
    style="background:#1565C0;color:#fff;border:none;padding:8px 14px;border-radius:8px;cursor:pointer;font-size:0.95rem;margin-right:8px;">
    🔊 {label}
  </button>
  <button type="button" id="btn_stop_{uid}"
    style="background:#E2E8F0;color:#1E293B;border:none;padding:8px 14px;border-radius:8px;cursor:pointer;font-size:0.95rem;">
    Parar leitura
  </button>
  <p style="font-size:0.8rem;color:#64748B;margin:8px 0 0 0;">
    Voz do navegador (português Brasil quando disponível). Requer clique para iniciar.
  </p>
</div>
<script>
(function() {{
  const txt = {safe_js};
  const speak = document.getElementById('btn_speak_{uid}');
  const stop = document.getElementById('btn_stop_{uid}');
  if (!window.speechSynthesis) {{
    speak.disabled = true;
    speak.textContent = 'Leitura por voz não suportada neste navegador';
    return;
  }}
  speak.addEventListener('click', function() {{
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(txt);
    u.lang = 'pt-BR';
    u.rate = 0.92;
    window.speechSynthesis.speak(u);
  }});
  stop.addEventListener('click', function() {{
    window.speechSynthesis.cancel();
  }});
}})();
</script>
"""
    components.html(html, height=118)


def bloco_texto_leitores_ecra(texto: str) -> str:
    return "##### Descrição para leitores de ecrã\n\n" + texto + "\n\n---\n"


def texto_home_completo_tts() -> str:
    from utils.castanhal_apresentacao import texto_tts_apresentacao

    base = (
        "Página inicial do painel sobre Castanhal, Pará. Trabalho de conclusão de curso com dados agregados do IBGE "
        "e projeções com redes neurais do notebook Arquivo TCC. "
    )
    ctx = texto_tts_apresentacao()
    resto = (
        " Seguem quatro métricas resumo e gráficos: população por censo; composição por cor ou raça em 2022; "
        "evolução do rendimento per capita."
    )
    return base + ctx + resto


TEXTO_HOME = texto_home_completo_tts()

TEXTO_DADOS = (
    "Página de dados e projeções com rede neural MLP. Tabela com indicadores que têm pelo menos dois censos entre 1991, 2000, 2010 e 2022. "
    "Separadores por tema. Em cada indicador: gráfico com censos, curva do modelo e projeções, mais texto explicativo."
)

TEXTO_PERGUNTAS = (
    "Perguntas ao assistente Gemini. Modo dados usa o contexto do notebook; modo livre para temas gerais. "
    "Campo de chat na base. Requer chave da API nos segredos da aplicação."
)

def _texto_info_tts() -> str:
    from utils.castanhal_apresentacao import texto_tts_apresentacao

    base = (
        "Informações sobre o TCC, autor Luan Evaristo de Melo Lindolfo, fluxo Colab e GitHub, diagrama e agradecimento. "
        "Este site inclui descrições em texto e botão opcional para ouvir a descrição de cada secção pela voz do navegador. "
    )
    return base + texto_tts_apresentacao()


TEXTO_INFO = _texto_info_tts()

TEXTO_DOWNLOAD = (
    "Transferência de relatórios: pré-visualização em tabela; botão CSV UTF-8 com indicadores e projeções; botão PDF resumo se disponível."
)

TEXTO_TRILHA = (
    "Trilha interativa Castanhal em dados: avança por salas temáticas, cada uma desbloqueia um gráfico ou resumo real dos censos. "
    "Usa os botões Avançar e Voltar. No fim, liga às páginas Dados e Download."
)
