"""
Textos de apresentação contextual sobre o município de Castanhal–PA (uso em Home, Info e TTS).
"""

from __future__ import annotations

# Tuplas (rótulo curto, texto)
BLOCOS_APRESENTACAO: list[tuple[str, str]] = [
    (
        "Origem",
        "O nome **Castanhal** deve-se à abundância de **castanheiras** na região.",
    ),
    (
        "Impulso urbano",
        "A localidade cresceu em torno da **Estrada de Ferro de Bragança** no **final do século XIX**.",
    ),
    (
        "Colonização",
        "A formação populacional foi marcada pela imigração de **cearenses** e **europeus**, com destaque para **espanhóis**.",
    ),
    (
        "Emancipação",
        "Tornou-se **município em 1932**, desmembrando-se de **Santa Izabel do Pará**.",
    ),
    (
        "Atualidade",
        "É conhecida como **«Cidade Modelo»** pela organização urbana e pela **força no comércio regional**.",
    ),
]


def markdown_apresentacao() -> str:
    """Markdown para exibição na interface."""
    linhas = ["### Castanhal em contexto histórico\n"]
    for titulo, texto in BLOCOS_APRESENTACAO:
        linhas.append(f"**{titulo}** — {texto}\n")
    return "\n".join(linhas)


def texto_tts_apresentacao() -> str:
    """Parágrafo corrido para síntese de voz."""
    partes = ["Contexto sobre Castanhal no Pará."]
    for titulo, texto in BLOCOS_APRESENTACAO:
        limpo = texto.replace("*", "").replace("**", "")
        partes.append(f"{titulo}: {limpo}")
    return " ".join(partes)
