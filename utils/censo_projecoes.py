"""
Séries históricas e metadados extraídos do notebook Arquivo_tcc.ipynb (IBGE comparativos).
Projeções reproduzem a abordagem MLPRegressor + StandardScaler do notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SerieCenso:
    """Uma série usada nos gráficos de rede neural do notebook."""

    id: str
    titulo: str
    area: str
    anos: tuple[int, ...]
    valores: tuple[float, ...]
    unidade: str
    descricao_curta: str
    o_que_foi_feito: str
    nota_prev: str


# Fonte: células do Arquivo_tcc.ipynb (arrays np.array explícitos)
SERIES: list[SerieCenso] = [
    SerieCenso(
        id="pop_total",
        titulo="População residente total",
        area="Demografia",
        anos=(1991, 2000, 2010, 2022),
        valores=(102_071, 134_496, 173_149, 192_256),
        unidade="habitantes",
        descricao_curta="Evolução do total de habitantes em Castanhal nos quatro censos com dados comparativos.",
        o_que_foi_feito="No notebook, uma rede neural (MLP) foi treinada sobre os quatro pontos censitários, com escalonamento dos anos e dos valores.",
        nota_prev="As previsões para 2030 e 2040 são extrapolações estatísticas; não substituem projeções oficiais do IBGE.",
    ),
    SerieCenso(
        id="rend_per_capita",
        titulo="Rendimento domiciliar mensal per capita",
        area="Renda",
        anos=(1991, 2000, 2010, 2022),
        valores=(160.54, 260.98, 467.32, 1055.05),
        unidade="R$",
        descricao_curta="Rendimento médio mensal per capita expresso em reais dos censos comparáveis.",
        o_que_foi_feito="Mesma família de modelos (MLP + scaler) aplicada à série de renda para traçar tendência e cenários futuros.",
        nota_prev="Valores nominais; comparação longitudinal deve considerar inflação e poder de compra.",
    ),
    SerieCenso(
        id="idhm_edu",
        titulo="IDHM Educação",
        area="Educação",
        anos=(1991, 2000, 2010),
        valores=(0.250, 0.404, 0.582),
        unidade="índice (0–1)",
        descricao_curta="Componente educação do IDH municipal (série com três censos no notebook).",
        o_que_foi_feito="Modelo neural com três observações; no notebook as projeções alcançam anos posteriores ao último censo observado.",
        nota_prev="IDHM usa metodologia PNUD; interpretar junto aos demais componentes do IDH.",
    ),
    SerieCenso(
        id="pop_urbana",
        titulo="População urbana",
        area="Território",
        anos=(2000, 2010, 2022),
        valores=(121_250, 147_662, 177_867),
        unidade="habitantes",
        descricao_curta="População em situação urbana conforme recortes censitários disponíveis.",
        o_que_foi_feito="Projeção MLP sobre três pontos para analisar tendência de urbanização.",
        nota_prev="Recortes de 2000–2022; não inclui 1991 nesta série.",
    ),
    SerieCenso(
        id="pop_rural",
        titulo="População rural",
        area="Território",
        anos=(2000, 2010, 2022),
        valores=(13_246, 25_487, 14_389),
        unidade="habitantes",
        descricao_curta="População em situação rural — útil para contrastar com o eixo urbano.",
        o_que_foi_feito="Modelagem análoga à população urbana, evidenciando oscilações entre censos.",
        nota_prev="Quedas ou saltos podem refletir mudanças de critério territorial e classificação.",
    ),
    SerieCenso(
        id="media_moradores",
        titulo="Média de moradores por domicílio",
        area="Domicílios",
        anos=(1991, 2010, 2022),
        valores=(4.2, 3.31, 3.01),
        unidade="pessoas/domicílio",
        descricao_curta="Tamanho médio dos domicílios; o valor de 1991 segue a convenção do notebook (referência regional).",
        o_que_foi_feito="Redes neurais sobre três pontos para suavizar tendência de redução média.",
        nota_prev="Indicador agregado; não confundir com tamanho de família nos quadros temáticos.",
    ),
    SerieCenso(
        id="total_domicilios",
        titulo="Total de domicílios recenseados",
        area="Domicílios",
        anos=(2010, 2022),
        valores=(52_769, 80_009),
        unidade="domicílios",
        descricao_curta="Quantidade de domicílios nos dois censos em que a série aparece no notebook.",
        o_que_foi_feito="Com apenas dois pontos, o MLP extrapola com forte incerteza — útil como exercício de tendência.",
        nota_prev="Projeções com 2 pontos são ilustrativas; intervalos de confiança seriam amplos.",
    ),
    SerieCenso(
        id="pop_parda",
        titulo="População parda",
        area="Cor ou raça",
        anos=(2010, 2022),
        valores=(119_139, 132_079),
        unidade="habitantes",
        descricao_curta="Contagem absoluta autodeclarada — componente majoritário em Castanhal.",
        o_que_foi_feito="Matriz 12 redes (ativação × solver) no notebook; aqui usamos um MLP representativo para curva e previsão.",
        nota_prev="Categorias de cor/raça dependem de autodeclaração e podem variar entre censos.",
    ),
    SerieCenso(
        id="pop_branca",
        titulo="População branca",
        area="Cor ou raça",
        anos=(2010, 2022),
        valores=(42_784, 42_881),
        unidade="habitantes",
        descricao_curta="Estabilidade relativa entre 2010 e 2022 no recorte do notebook.",
        o_que_foi_feito="Modelagem neural para suavizar e projetar cenários futuros.",
        nota_prev="Pequenas variações absolutas podem mascarar mudanças proporcionais ao total.",
    ),
    SerieCenso(
        id="pop_preta",
        titulo="População preta",
        area="Cor ou raça",
        anos=(2010, 2022),
        valores=(9_126, 16_429),
        unidade="habitantes",
        descricao_curta="Crescimento expressivo entre os dois censos no conjunto de dados do TCC.",
        o_que_foi_feito="Projeção MLP a partir de dois pontos históricos.",
        nota_prev="Interpretar com cautela políticas públicas antirracistas e qualidade dos dados.",
    ),
    SerieCenso(
        id="pop_amarela",
        titulo="População amarela",
        area="Cor ou raça",
        anos=(2010, 2022),
        valores=(1_944, 719),
        unidade="habitantes",
        descricao_curta="Queda absoluta entre censos no arquivo do notebook — refletir categorias IBGE.",
        o_que_foi_feito="Extrapolação com MLP; valores pequenos amplificam sensibilidade do modelo.",
        nota_prev="Mudanças metodológicas e de autodeclaração afetam séries de cor/raça.",
    ),
    SerieCenso(
        id="pop_indigena",
        titulo="População indígena",
        area="Cor ou raça",
        anos=(2010, 2022),
        valores=(156, 144),
        unidade="habitantes",
        descricao_curta="População absoluta indígena no recorte municipal do notebook.",
        o_que_foi_feito="Modelo neural ilustrativo sobre dois pontos.",
        nota_prev="Populações pequenas exigem interpretação cuidadosa e respeito a políticas indigenistas.",
    ),
    SerieCenso(
        id="prop_indigena",
        titulo="Proporção indígena na população de Castanhal",
        area="Populações tradicionais",
        anos=(2010, 2022),
        valores=(0.09, 0.07),
        unidade="proporção",
        descricao_curta="Participação relativa estimada no notebook (valores em proporção, não percentual).",
        o_que_foi_feito="Série com dois pontos para análise de tendência proporcional.",
        nota_prev="Consistência com numeradores e denominadores municipais deve ser validada em fontes oficiais.",
    ),
    SerieCenso(
        id="registro_cartorio",
        titulo="Registro de nascimento em cartório (indicador agregado)",
        area="Populações tradicionais",
        anos=(2010, 2022),
        valores=(100.0, 90.0),
        unidade="percentual",
        descricao_curta="Indicador derivado da tabela de indígenas no notebook (escala 0–100).",
        o_que_foi_feito="Projeção MLP sobre percentuais — interpretar como exercício de tendência.",
        nota_prev="Percentuais agregados não substituem análises de políticas de registro civil.",
    ),
    SerieCenso(
        id="sem_registro",
        titulo="Sem registro civil de nascimento (indicador agregado)",
        area="Populações tradicionais",
        anos=(2010, 2022),
        valores=(0.0, 10.0),
        unidade="percentual",
        descricao_curta="Complementar ao indicador de registro em cartório no mesmo recorte temático.",
        o_que_foi_feito="Extrapolação neural a partir de dois pontos.",
        nota_prev="Uso acadêmico; políticas públicas exigem validação com órgãos competentes.",
    ),
]


def _anos_futuros_padrao(anos: tuple[int, ...]) -> tuple[int, int]:
    """Define anos futuros coerentes com o notebook (variações por comprimento de série)."""
    n = len(anos)
    if n >= 4:
        return (2030, 2040)
    if n == 3 and anos[0] == 1991 and anos[-1] == 2010:  # IDHM
        return (2022, 2030)
    if n == 3:
        return (2030, 2040)
    return (2030, 2040)


@lru_cache(maxsize=128)
def projetar_mlp(
    anos: tuple[int, ...],
    valores: tuple[float, ...],
    anos_futuros: tuple[int, int] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Treina MLPRegressor (10,10), solver lbfgs — alinhado ao notebook Arquivo_tcc.ipynb.
    Retorna séries para gráfico e previsões pontuais.
    """
    if anos_futuros is None:
        anos_futuros = _anos_futuros_padrao(anos)
    X = np.array(anos, dtype=float).reshape(-1, 1)
    y = np.array(valores, dtype=float).reshape(-1, 1)
    sx = StandardScaler()
    sy = StandardScaler()
    Xs = sx.fit_transform(X)
    ys = sy.fit_transform(y).ravel()
    mlp = MLPRegressor(
        hidden_layer_sizes=(10, 10),
        max_iter=5000,
        solver="lbfgs",
        random_state=random_state,
    )
    mlp.fit(Xs, ys)

    ano_min, ano_max = min(anos), max(anos)
    fut_max = max(anos_futuros)
    anos_curva = np.linspace(ano_min - 0.5, max(ano_max, fut_max) + 0.5, 120).reshape(-1, 1)
    curva_scaled = mlp.predict(sx.transform(anos_curva)).reshape(-1, 1)
    curva = sy.inverse_transform(curva_scaled).ravel()

    preds = []
    for af in anos_futuros:
        p = sy.inverse_transform(
            mlp.predict(sx.transform(np.array([[af]]))).reshape(-1, 1)
        )[0, 0]
        preds.append((int(af), float(p)))

    return {
        "anos_hist": list(anos),
        "val_hist": list(valores),
        "anos_curva": anos_curva.ravel(),
        "val_curva": curva,
        "previsoes": preds,
        "anos_futuros": anos_futuros,
    }


def dataframe_relatorio_completo() -> pd.DataFrame:
    """Uma linha por série: histórico + previsões consolidadas para CSV/PDF."""
    linhas = []
    for s in SERIES:
        pr = projetar_mlp(s.anos, s.valores)
        row = {
            "area": s.area,
            "indicador": s.titulo,
            "unidade": s.unidade,
            "censos_utilizados": ",".join(str(a) for a in s.anos),
            "n_censos": len(s.anos),
        }
        for i, a in enumerate(s.anos):
            row[f"valor_{a}"] = s.valores[i]
        for (af, pv) in pr["previsoes"]:
            row[f"previsto_{af}"] = round(pv, 4)
        linhas.append(row)
    return pd.DataFrame(linhas)


def series_com_pelo_menos_2_censos_no_periodo() -> list[SerieCenso]:
    """Indicadores cujo histórico cruza ao menos dois censos entre 1991–2022 (incluindo recortes parciais)."""
    censos_alvo = {1991, 2000, 2010, 2022}
    out = []
    for s in SERIES:
        usados = set(s.anos) & censos_alvo
        if len(usados) >= 2:
            out.append(s)
    return out
