"""
Configuração da «trilha» interativa — estilo dungeon/streamlit: passos com narrativa + dados reais.
"""

from __future__ import annotations

from typing import Any

# Cada passo: id estável, títulos, markdown narrativo, id da série em censo_projecoes (ou None no último)
TRILHA_PASSOS: list[dict[str, Any]] = [
    {
        "id": "entrada",
        "titulo": "🏛️ Entrada — Bem-vindo a Castanhal",
        "narrativa": (
            "Você entra na **trilha dos dados censitários** de Castanhal, Pará. "
            "Este é um TCC que transforma tabelas do IBGE em história e projeções. "
            "Cada sala revela um recorte real do notebook *Arquivo_tcc.ipynb*. "
            "Clique em **Avançar** para a primeira descoberta."
        ),
        "serie_id": None,
    },
    {
        "id": "populacao",
        "titulo": "📍 Sala 1 — A cidade cresceu",
        "narrativa": (
            "A população residente passou de cerca de **102 mil** habitantes em 1991 para **192 mil** em 2022. "
            "O gráfico abaixo mostra a série e a tendência do modelo neural (MLP) usado no estudo."
        ),
        "serie_id": "pop_total",
    },
    {
        "id": "renda",
        "titulo": "💰 Sala 2 — Renda por habitante",
        "narrativa": (
            "O **rendimento domiciliar mensal per capita** subiu ao longo dos censos (valores nominais no gráfico). "
            "As projeções são ilustrativas — não substituem análises oficiais."
        ),
        "serie_id": "rend_per_capita",
    },
    {
        "id": "territorio",
        "titulo": "🏙️ Sala 3 — Urbano e rural",
        "narrativa": (
            "Castanhal é majoritariamente **urbana**. Compare população urbana e rural entre 2000 e 2022 — "
            "útil para políticas de mobilidade e saneamento."
        ),
        "serie_id": "pop_urbana",
    },
    {
        "id": "cor_raca",
        "titulo": "🌈 Sala 4 — Cor ou raça (2010–2022)",
        "narrativa": (
            "A população **parda** é a maior parcela absoluta; os gráficos do TCC exploram vários grupos. "
            "Interpretação responsável: dados agregados e autodeclaração censitária."
        ),
        "serie_id": "pop_parda",
    },
    {
        "id": "tesouro",
        "titulo": "🗝️ Sala final — Tesouro de dados",
        "narrativa": (
            "Você concluiu a trilha introdutória! **Próximos passos:** abra a página **Dados** para explorar "
            "todas as séries e abas temáticas; use **Download** para CSV/PDF; ou **Perguntas** para conversar com a IA sobre os indicadores."
        ),
        "serie_id": None,
    },
]


def serie_por_id(sid: str):
    from utils.censo_projecoes import SERIES

    for s in SERIES:
        if s.id == sid:
            return s
    return None


def texto_tts_passo(passo: dict[str, Any], indice: int, total: int) -> str:
    """Texto corrido para TTS no passo atual."""
    tit = passo.get("titulo", "")
    nar = passo.get("narrativa", "").replace("*", "").replace("**", "")
    return (
        f"Trilha Castanhal em dados. Passo {indice + 1} de {total}. {tit}. {nar}"
    )
