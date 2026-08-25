# -*- coding: utf-8 -*-
"""
TCC — Projeções Censitárias com Redes Neurais (versão CORINGA)
Edite apenas o bloco CONFIGURAÇÃO para trocar de município.

Diferença em relação à versão anterior:
- Não existe mais uma lista manual `INDICADORES`. O script percorre
  TODAS as linhas de TODOS os CSVs carregados, extrai os valores das
  colunas de censo e classifica cada indicador automaticamente em:
    Grupo 1 -> 4 censos com dado válido
    Grupo 2 -> 3 censos com dado válido
    Grupo 3 -> 2 censos com dado válido
    (descartado -> 0 ou 1 censo com dado válido)
- `OVERRIDES` (opcional) permite ajustar ativação/solver/format/y_lim
  de um indicador específico, sem precisar redeclarar tudo.
"""

from google.colab import drive
drive.mount('/content/drive')

!pip -q install streamlit plotly yellowbrick numpy statsmodels fpdf2 pyarrow

import re
import unicodedata
import itertools
import warnings
import os
import shutil
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.exceptions import ConvergenceWarning
from scipy.optimize import curve_fit
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore', category=ConvergenceWarning)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║              BLOCO DE CONFIGURAÇÃO — EDITE APENAS AQUI                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

MUNICIPIO = "Castanhal"

# Pasta raiz no Google Drive com os CSVs do município
BASE_DRIVE = "/content/drive/MyDrive/dados_ibge/Dados comparativos 90, 00, 10 e 22 - Castanhal"

# Mapa: chave interna → caminho relativo ao BASE_DRIVE
ARQUIVOS = {
    "composicaofamiliar":    "Composição familiar/composição familiar_2.csv",
    "deslocamento":          "Deslocamento para trabalho e estudo/Deslocamento.csv",
    "domicilios":            "Domicílios/Raio-X do Lar Urbano.csv",
    "educacao":              "Educação/educacao.csv",
    "entornodomicilios":     "Entorno dos domicílios/entorno.csv",
    "familiasenupcialidade": "Famílias e Nupcialidade/Familias_e_Nupcialidade.csv",
    "favelasecomunidade":    "Favelas e comunidades urbanas/Favelas.csv",
    "indigenas":             "Indígenas/indigenas.csv",
    "populacao":             "População/populacao.csv",
    "quilombolas":           "Quilombolas/Quilombolas.csv",
    "religiao":              "Religião/religiao.csv",
    "trabalhoerendimento":   "Trabalho e rendimento/Trabalho_e_Rendimento.csv",
}

# Nome da coluna identificadora nos CSVs
COL_INDICADOR = "Indicador / Categoria"

# Colunas de ano (ajuste se os cabeçalhos forem diferentes nos seus CSVs)
COL_1991 = "Censo 1991"
COL_2000 = "Censo 2000"
COL_2010 = "Censo 2010"
COL_2022 = "Censo 2022"
COLUNAS_CENSO = [COL_1991, COL_2000, COL_2010, COL_2022]

# ── OVERRIDES (opcional) ──────────────────────────────────────────────────
# Ajustes finos por indicador, encontrados via explorar_combinacoes_rna().
# A chave é gerada automaticamente como "<df_key>__<slug_do_indicador>"
# (o próprio script imprime a chave de cada indicador descoberto, use-a aqui).
# Nada aqui é obrigatório — sem overrides, todo indicador usa
# ativacao='relu', solver='lbfgs' por padrão.
OVERRIDES = {
    # exemplo:
    # "populacao__populacao_residente_total": dict(ativacao="relu", solver="lbfgs"),
    # "educacao__idhm_educacao_indice_de_desenv_humano": dict(
    #     ativacao="identity", solver="lbfgs", format_str="{:.3f}", y_lim=(0.1, 1.1),
    # ),
}

# Parâmetros globais dos modelos
RANDOM_STATE  = 42
HIDDEN_LAYERS = (10, 10)

# Anos futuros padrão para projeção
ANOS_FUTUROS_PADRAO = [2030, 2040]

# Pasta de saída local no Colab
DIR_PROCESSED = '/content/dados_processados'

# GitHub
GITHUB_USUARIO = "LuanLindolfo"
GITHUB_REPO    = "tcc"
GITHUB_BRANCH  = "main"
GITHUB_EMAIL   = "luan.lindolfo1211@gmail.com"
# Pasta dentro do repositório onde os dados serão salvos
# Castanhal → "data" | Belém → "data_belem" | nova cidade → "data_<cidade>"
GITHUB_PASTA   = "data"

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     FIM DO BLOCO DE CONFIGURAÇÃO                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════════
# 0) DIAGNÓSTICO DE CAMINHO (rode isso primeiro se der "Não encontrado")
# ═══════════════════════════════════════════════════════════════════════════

def diagnosticar_base_drive():
    """
    Lista o que realmente existe no Drive perto de BASE_DRIVE, para você
    comparar com o nome configurado e corrigir acentos/espaços/maiúsculas.
    """
    raiz = os.path.dirname(BASE_DRIVE)  # ex: /content/drive/MyDrive/dados_ibge
    print(f"🔎 Verificando: {raiz}")
    if not os.path.isdir(raiz):
        print(f"  ❌ Essa pasta-raiz nem existe. Conteúdo de 'dados_ibge' pai a pai:")
        pai = os.path.dirname(raiz)
        if os.path.isdir(pai):
            for item in sorted(os.listdir(pai)):
                print(f"    - {item!r}")
        return

    print(f"  Pastas de município encontradas em '{raiz}':")
    for item in sorted(os.listdir(raiz)):
        marca = "✅" if os.path.join(raiz, item) == BASE_DRIVE else "  "
        print(f"    {marca} {item!r}")

    if os.path.isdir(BASE_DRIVE):
        print(f"\n  BASE_DRIVE existe ✅ — listando subpastas dentro dele:")
        for item in sorted(os.listdir(BASE_DRIVE)):
            print(f"    - {item!r}")
    else:
        print(f"\n  ❌ BASE_DRIVE configurado não bate com nenhuma pasta acima.")
        print(f"     Configurado: {BASE_DRIVE!r}")


diagnosticar_base_drive()


# ═══════════════════════════════════════════════════════════════════════════
# 1) CARGA DOS DADOS
# ═══════════════════════════════════════════════════════════════════════════

def _ler_csv_robusto(caminho: str) -> pd.DataFrame:
    """
    Tenta ler o CSV de forma tolerante:
      1) vírgula (padrão)
      2) ponto-e-vírgula (comum em CSV exportado de Excel BR)
      3) detecção automática do separador (engine='python', sep=None)
      4) por fim, tolera linhas malformadas (on_bad_lines='skip')
    Tenta também utf-8 e latin-1 em cada etapa.
    """
    tentativas = [
        dict(sep=',', encoding='utf-8'),
        dict(sep=';', encoding='utf-8'),
        dict(sep=',', encoding='latin-1'),
        dict(sep=';', encoding='latin-1'),
        dict(sep=None, engine='python', encoding='utf-8'),
        dict(sep=None, engine='python', encoding='latin-1'),
    ]
    for kw in tentativas:
        try:
            df = pd.read_csv(caminho, **kw)
            if df.shape[1] > 1:  # descarta leituras que vieram como 1 coluna só (sep errado)
                return df
        except Exception:
            continue

    # última tentativa: pula linhas malformadas em vez de travar
    for kw in tentativas:
        try:
            df = pd.read_csv(caminho, on_bad_lines='skip', **kw)
            if df.shape[1] > 1:
                print(f"    ⚠️  lido tolerando linhas malformadas ({kw})")
                return df
        except Exception:
            continue

    raise ValueError(f"Não foi possível ler {caminho} com nenhuma combinação de sep/encoding.")


dados = {}
for nome, caminho_relativo in ARQUIVOS.items():
    caminho_completo = f"{BASE_DRIVE}/{caminho_relativo}"
    try:
        dados[nome] = _ler_csv_robusto(caminho_completo)
        print(f"✅ {nome}  ({dados[nome].shape[0]} linhas, {dados[nome].shape[1]} colunas)")
    except FileNotFoundError:
        print(f"⚠️  Não encontrado, pulando: {caminho_completo}")
    except ValueError as e:
        print(f"❌ Falha ao ler {nome}: {e}")

for nome, df in dados.items():
    globals()[nome] = df

print(f"\n{len(dados)} arquivo(s) carregado(s).")


# ═══════════════════════════════════════════════════════════════════════════
# 2) FUNÇÕES AUXILIARES DE PARSING
# ═══════════════════════════════════════════════════════════════════════════

def _parse_numero(raw) -> float:
    """
    Extrai o primeiro número de uma célula bruta, tratando formato BR:
    ponto como milhar, vírgula como decimal.
    Ex: '1.393.399 hab.' → 1393399.0 | '3,52' → 3.52 | NaN → NaN
    """
    if pd.isna(raw):
        return np.nan
    texto = str(raw).replace('\xa0', '').strip()
    match = re.search(r'[\d.,]+', texto)
    if not match:
        return np.nan
    s = match.group()
    if ',' in s and '.' in s:
        s = s.replace('.', '').replace(',', '.')
    elif ',' in s:
        s = s.replace(',', '.')
    elif s.count('.') > 1:
        s = s.replace('.', '')
    try:
        return float(s)
    except ValueError:
        return np.nan


def _slugify(texto: str, maxlen: int = 45) -> str:
    """Transforma um texto livre em identificador seguro (sem acento/espaço)."""
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')
    texto = texto.lower().strip()
    texto = re.sub(r'[^a-z0-9]+', '_', texto).strip('_')
    return texto[:maxlen] or 'indicador'


def _secao(titulo: str):
    display(Markdown(f"### {titulo}"))


# ═══════════════════════════════════════════════════════════════════════════
# 3) DESCOBERTA E AGRUPAMENTO AUTOMÁTICO DOS INDICADORES  ◄── NÚCLEO CORINGA
# ═══════════════════════════════════════════════════════════════════════════

def descobrir_e_agrupar_indicadores(dados: dict) -> dict:
    """
    Percorre TODOS os DataFrames carregados, linha a linha:
      1) lê o texto do indicador (coluna COL_INDICADOR)
      2) extrai o valor de cada coluna de censo presente no CSV
      3) conta quantos censos têm valor numérico válido (não-NaN)
      4) classifica automaticamente:
           4 censos válidos -> "Grupo 1 — 4 censos"
           3 censos válidos -> "Grupo 2 — 3 censos"
           2 censos válidos -> "Grupo 3 — 2 censos"
           0 ou 1 censo     -> descartado (não entra no dicionário)

    Retorna um dicionário {chave: config} pronto para treinar/plotar,
    sem necessidade de nenhuma lista manual de indicadores.
    """
    config = {}
    chaves_usadas = set()
    contagem_grupo = {"Grupo 1 — 4 censos": 0, "Grupo 2 — 3 censos": 0,
                       "Grupo 3 — 2 censos": 0, "descartado (≤1 censo)": 0}

    for df_key, df in dados.items():
        if COL_INDICADOR not in df.columns:
            print(f"⚠️  '{df_key}': coluna '{COL_INDICADOR}' ausente — arquivo inteiro ignorado.")
            continue

        colunas_presentes = [c for c in COLUNAS_CENSO if c in df.columns]
        if not colunas_presentes:
            print(f"⚠️  '{df_key}': nenhuma coluna de censo reconhecida — arquivo ignorado.")
            continue

        for _, row in df.iterrows():
            indicador_txt = str(row[COL_INDICADOR]).strip()
            if not indicador_txt or indicador_txt.lower() in ('nan', 'none', ''):
                continue

            pares = []
            for col in colunas_presentes:
                valor = _parse_numero(row[col])
                if not np.isnan(valor):
                    ano = int(re.search(r'\d{4}', col).group())
                    pares.append((ano, valor))

            n_censos = len(pares)

            if n_censos >= 4:
                grupo = "Grupo 1 — 4 censos"
            elif n_censos == 3:
                grupo = "Grupo 2 — 3 censos"
            elif n_censos == 2:
                grupo = "Grupo 3 — 2 censos"
            else:
                contagem_grupo["descartado (≤1 censo)"] += 1
                continue

            pares.sort(key=lambda p: p[0])
            anos    = [a for a, _ in pares]
            valores = [v for _, v in pares]

            base_chave = f"{df_key}__{_slugify(indicador_txt)}"
            chave = base_chave
            n = 1
            while chave in chaves_usadas:
                n += 1
                chave = f"{base_chave}_{n}"
            chaves_usadas.add(chave)
            contagem_grupo[grupo] += 1

            ov = OVERRIDES.get(chave, {})

            config[chave] = {
                "titulo":       indicador_txt,
                "df_key":       df_key,
                "anos":         anos,
                "valores":      valores,
                "n_censos":     n_censos,
                "grupo":        grupo,
                "ylabel":       ov.get("ylabel", "Valor"),
                "ativacao":     ov.get("ativacao", "relu"),
                "solver":       ov.get("solver", "lbfgs"),
                "format_str":   ov.get("format_str", "{:.2f}"),
                "y_lim":        ov.get("y_lim"),
                "future_years": ov.get("future_years", ANOS_FUTUROS_PADRAO),
                "curve_range":  ov.get("curve_range", (min(anos) - 5, max(anos) + 16)),
            }

    print("\n📊 Resumo do agrupamento automático:")
    for grupo, qtd in contagem_grupo.items():
        print(f"  {grupo}: {qtd} indicador(es)")

    return config


SERIES_CONFIG = descobrir_e_agrupar_indicadores(dados)


def _achar_chave(config: dict, df_key: str = None, contem: str = "") -> str:
    """Localiza a chave de um indicador por df_key + trecho do título (busca livre)."""
    for chave, cfg in config.items():
        if df_key and cfg["df_key"] != df_key:
            continue
        if contem.lower() in cfg["titulo"].lower():
            return chave
    return None


# Exibição dos indicadores agrupados (equivalente à antiga seção 3)
for grupo_nome in ["Grupo 1 — 4 censos", "Grupo 2 — 3 censos", "Grupo 3 — 2 censos"]:
    _secao(f"{MUNICIPIO} — {grupo_nome}")
    linhas_grupo = [
        {"indicador": cfg["titulo"], "df": cfg["df_key"],
         **{str(a): v for a, v in zip(cfg["anos"], cfg["valores"])}}
        for cfg in SERIES_CONFIG.values() if cfg["grupo"] == grupo_nome
    ]
    if linhas_grupo:
        display(pd.DataFrame(linhas_grupo))
    else:
        print("  (nenhum indicador nesse grupo)")


# ═══════════════════════════════════════════════════════════════════════════
# 4) MODELOS — FUNÇÕES GENÉRICAS
# ═══════════════════════════════════════════════════════════════════════════

def _treinar_mlp(anos, valores, ativacao, solver, anos_alvo,
                 max_iter=5000, **kwargs):
    X = np.array(anos).reshape(-1, 1)
    y = np.array(valores).reshape(-1, 1)
    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X)
    ys = sy.fit_transform(y)
    rs = kwargs.pop("random_state", RANDOM_STATE)
    m = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS, activation=ativacao,
        solver=solver, max_iter=max_iter, random_state=rs, **kwargs,
    )
    m.fit(Xs, ys.ravel())
    return sy.inverse_transform(
        m.predict(sx.transform(
            np.array(anos_alvo).reshape(-1, 1)
        )).reshape(-1, 1)
    )


def explorar_combinacoes_rna(chave: str, config=None):
    """Grid 4×3 (ativação × solver) para qualquer indicador em SERIES_CONFIG."""
    if config is None:
        config = SERIES_CONFIG
    cfg = config[chave]
    anos, valores = cfg["anos"], cfg["valores"]

    if len(anos) < 2:
        print(f"⚠️  '{chave}' tem menos de 2 pontos — não é possível treinar.")
        return

    anos_curva  = np.arange(*cfg["curve_range"]).reshape(-1, 1)
    anos_fut    = np.array(cfg["future_years"])
    combinacoes = list(itertools.product(
        ['identity', 'logistic', 'tanh', 'relu'],
        ['lbfgs', 'sgd', 'adam'],
    ))

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(f"{cfg['titulo']} — {MUNICIPIO} ({cfg['grupo']})\nMatriz de RNAs",
                 fontsize=18, y=0.92)
    print(f"Treinando 12 RNAs para '{chave}'...")

    for ax, (ativ, solver) in zip(axes.flatten(), combinacoes):
        tend = _treinar_mlp(anos, valores, ativ, solver, anos_curva.ravel())
        prev = _treinar_mlp(anos, valores, ativ, solver, anos_fut)
        ax.scatter(anos, valores, color='royalblue', s=100, zorder=5, label='Dados Reais')
        ax.plot(anos_curva, tend, color='indigo', linewidth=2.5)
        ax.scatter(anos_fut, prev, color='red', s=80, zorder=5)
        fmt = cfg["format_str"]
        ax.set_title(f"'{ativ}' | '{solver}'\nPrev {anos_fut[-1]}: {fmt.format(prev[-1][0])}",
                     fontsize=11)
        ax.set_ylabel(cfg["ylabel"])
        if cfg["y_lim"]:
            ax.set_ylim(*cfg["y_lim"])
        ax.grid(True, linestyle=':', alpha=0.7)

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.show()


def plot_final_model(chave: str, config=None):
    """Gráfico final com a ativação/solver definida (ou default) do indicador."""
    if config is None:
        config = SERIES_CONFIG
    cfg = config[chave]
    anos, valores = cfg["anos"], cfg["valores"]
    ativ, solver  = cfg["ativacao"], cfg["solver"]
    fmt           = cfg["format_str"]

    if len(anos) < 2:
        print(f"⚠️  '{chave}' tem menos de 2 pontos — gráfico ignorado.")
        return

    X  = np.array(anos).reshape(-1, 1)
    y  = np.array(valores).reshape(-1, 1)
    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X)
    ys = sy.fit_transform(y)

    m = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS, activation=ativ,
        solver=solver, max_iter=5000, random_state=RANDOM_STATE,
    )
    m.fit(Xs, ys.ravel())

    Xc = np.arange(*cfg["curve_range"]).reshape(-1, 1)
    yc = sy.inverse_transform(m.predict(sx.transform(Xc)).reshape(-1, 1))
    Xf = np.array(cfg["future_years"]).reshape(-1, 1)
    yf = sy.inverse_transform(m.predict(sx.transform(Xf)).reshape(-1, 1))

    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, color='blue', s=100, zorder=5, label='Dados Reais (IBGE)')
    plt.plot(Xc, yc, color='indigo', linewidth=2.5, label='Curva RNA')
    plt.scatter(Xf, yf, color='red', s=100, zorder=5, label='Projeção')
    plt.title(f"{cfg['titulo']} — {MUNICIPIO} ({cfg['grupo']})\n"
              f"Ativação: '{ativ}' | Solver: '{solver}'", fontsize=13, pad=15)
    for i, ano in enumerate(cfg["future_years"]):
        plt.annotate(f"{ano}: {fmt.format(yf[i][0])}", (Xf[i], yf[i]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=10, fontweight='bold')
    plt.ylabel(cfg["ylabel"])
    plt.xlabel('Ano')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    if cfg["y_lim"]:
        plt.ylim(*cfg["y_lim"])
    if np.max(y) > 1000:
        plt.ticklabel_format(style='plain', axis='y')
    plt.tight_layout()
    plt.show()


def gerar_todos_os_graficos_finais(config=None, grupo: str = None):
    """
    Gera o gráfico final de todos os indicadores em SERIES_CONFIG.
    Se `grupo` for informado (ex: "Grupo 1 — 4 censos"), filtra só esse grupo.
    """
    if config is None:
        config = SERIES_CONFIG
    for chave, cfg in config.items():
        if grupo and cfg["grupo"] != grupo:
            continue
        plot_final_model(chave, config)


# Descomente para gerar tudo de uma vez (pode ser MUITOS gráficos,
# já que agora todo indicador do CSV com 2+ censos entra automaticamente):
# gerar_todos_os_graficos_finais()

# Ou gere por grupo, um de cada vez:
# gerar_todos_os_graficos_finais(grupo="Grupo 1 — 4 censos")
# gerar_todos_os_graficos_finais(grupo="Grupo 2 — 3 censos")
# gerar_todos_os_graficos_finais(grupo="Grupo 3 — 2 censos")


# ═══════════════════════════════════════════════════════════════════════════
# 5) MODELOS ALTERNATIVOS DE POPULAÇÃO
#    (localiza "população total" dinamicamente dentro do SERIES_CONFIG)
# ═══════════════════════════════════════════════════════════════════════════

_pop_chave = (
    _achar_chave(SERIES_CONFIG, df_key="populacao", contem="população residente total")
    or _achar_chave(SERIES_CONFIG, df_key="populacao", contem="população total")
)
_pop_cfg     = SERIES_CONFIG.get(_pop_chave, {})
ANOS_POP     = _pop_cfg.get("anos", [])
POPULACAO    = _pop_cfg.get("valores", [])
_ANOS_FUT    = _pop_cfg.get("future_years", ANOS_FUTUROS_PADRAO)
_CURVE_RANGE = _pop_cfg.get("curve_range", (1990, 2045))
CURVA_POP    = np.arange(*_CURVE_RANGE)


def _pop_valida() -> bool:
    if not ANOS_POP or not POPULACAO:
        print("⚠️  Indicador de 'população total' não encontrado em SERIES_CONFIG.")
        return False
    if len(ANOS_POP) < 2:
        print("⚠️  'população total' tem menos de 2 censos válidos — modelos ignorados.")
        return False
    return True


def _plot_comparacao(anos, valores, anos_curva, tend, anos_fut, prev,
                     titulo, cor_curva, label_curva):
    plt.figure(figsize=(10, 6))
    plt.scatter(anos, valores, color='blue', s=100, label='Dados Reais (IBGE)', zorder=5)
    plt.plot(anos_curva, tend, color=cor_curva, linewidth=2, label=label_curva)
    plt.scatter(anos_fut, prev, color='red', s=100, label='Projeção', zorder=5)
    plt.title(titulo, fontsize=14)
    plt.xlabel('Ano')
    plt.ylabel('Habitantes')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.ticklabel_format(style='plain', axis='y')
    plt.show()


def modelo_rna_simples():
    if not _pop_valida(): return
    tend = _treinar_mlp(ANOS_POP, POPULACAO, 'relu', 'lbfgs', CURVA_POP)
    prev = _treinar_mlp(ANOS_POP, POPULACAO, 'relu', 'lbfgs', _ANOS_FUT)
    print("--- RNA SIMPLES ---")
    for a, p in zip(_ANOS_FUT, prev):
        print(f"  {a}: {int(p[0])} hab.")
    _plot_comparacao(ANOS_POP, POPULACAO, CURVA_POP, tend, _ANOS_FUT, prev,
                     f'População — {MUNICIPIO} (RNA)', 'purple', 'Curva RNA')


def modelo_polinomial(grau=2):
    if not _pop_valida(): return
    X    = np.array(ANOS_POP).reshape(-1, 1)
    poly = PolynomialFeatures(degree=grau)
    m    = LinearRegression().fit(poly.fit_transform(X), POPULACAO)
    prev = m.predict(poly.transform(np.array(_ANOS_FUT).reshape(-1, 1)))
    tend = m.predict(poly.transform(CURVA_POP.reshape(-1, 1)))
    print("--- POLINOMIAL ---")
    for a, p in zip(_ANOS_FUT, prev):
        print(f"  {a}: {int(p)} hab.")
    _plot_comparacao(ANOS_POP, POPULACAO, CURVA_POP, tend, _ANOS_FUT, prev,
                     f'População — {MUNICIPIO} (Polinomial grau {grau})', 'green', f'Grau {grau}')


def modelo_logistico(chute=(250000, 0.1, 2000)):
    if not _pop_valida(): return
    def _fn(t, K, r, t0):
        return K / (1 + np.exp(-r * (t - t0)))
    params, _ = curve_fit(_fn, ANOS_POP, POPULACAO, p0=chute, maxfev=10000)
    K, r, t0  = params
    prev = _fn(np.array(_ANOS_FUT), K, r, t0)
    print(f"--- LOGÍSTICO --- Teto K = {int(K)} hab.")
    for a, p in zip(_ANOS_FUT, prev):
        print(f"  {a}: {int(p)} hab.")


def modelo_svr():
    if not _pop_valida(): return
    X  = np.array(ANOS_POP).reshape(-1, 1)
    y  = np.array(POPULACAO).reshape(-1, 1)
    sx, sy = StandardScaler(), StandardScaler()
    m  = SVR(kernel='poly', degree=2, C=100.0).fit(
        sx.fit_transform(X), sy.fit_transform(y).ravel()
    )
    prev = sy.inverse_transform(
        m.predict(sx.transform(np.array(_ANOS_FUT).reshape(-1, 1))).reshape(-1, 1)
    )
    tend = sy.inverse_transform(
        m.predict(sx.transform(CURVA_POP.reshape(-1, 1))).reshape(-1, 1)
    )
    print("--- SVR ---")
    for a, p in zip(_ANOS_FUT, prev):
        print(f"  {a}: {int(p[0])} hab.")
    _plot_comparacao(ANOS_POP, POPULACAO, CURVA_POP, tend, _ANOS_FUT, prev,
                     f'População — {MUNICIPIO} (SVR)', 'magenta', 'SVR Poly')


def modelo_arima(order=(1, 1, 0)):
    if not _pop_valida(): return
    res      = ARIMA(np.array(POPULACAO, dtype=float), order=order, trend='t').fit()
    prev     = res.forecast(steps=2)
    anos_fut = [ANOS_POP[-1] + 10, ANOS_POP[-1] + 20]
    print("--- ARIMA ---")
    for a, p in zip(anos_fut, prev):
        print(f"  ~{a}: {int(p)} hab.")


def modelo_ensemble_rna(n=50):
    if not _pop_valida(): return
    print(f"--- ENSEMBLE {n} RNAs ---  aguarde...")
    lp, lt = [], []
    for i in range(n):
        lp.append(_treinar_mlp(ANOS_POP, POPULACAO, 'relu', 'lbfgs', _ANOS_FUT,
                               max_iter=2000, alpha=0.5, random_state=i))
        lt.append(_treinar_mlp(ANOS_POP, POPULACAO, 'relu', 'lbfgs', CURVA_POP,
                               max_iter=2000, alpha=0.5, random_state=i))
    prev = np.mean(lp, axis=0)
    tend = np.mean(lt, axis=0)
    for a, p in zip(_ANOS_FUT, prev):
        print(f"  {a}: {int(p[0])} hab.")
    plt.figure(figsize=(10, 6))
    for t in lt:
        plt.plot(CURVA_POP, t, color='purple', alpha=0.05)
    plt.scatter(ANOS_POP, POPULACAO, color='blue', s=100, zorder=5, label='Dados Reais (IBGE)')
    plt.plot(CURVA_POP, tend, color='indigo', linewidth=3, label='Consenso do Comitê')
    plt.scatter(_ANOS_FUT, prev, color='red', s=100, zorder=5, label='Projeção')
    plt.title(f'População — {MUNICIPIO} (Ensemble {n} RNAs)', fontsize=14)
    plt.xlabel('Ano')
    plt.ylabel('Habitantes')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.ticklabel_format(style='plain', axis='y')
    plt.show()


modelo_rna_simples()
modelo_polinomial()
modelo_logistico()
modelo_svr()
modelo_arima()
modelo_ensemble_rna()


# ═══════════════════════════════════════════════════════════════════════════
# 6) EXPORTAÇÃO CSV / PARQUET
#    Nome do arquivo segue o padrão: indicadores_<municipio>_tratados.csv
# ═══════════════════════════════════════════════════════════════════════════

os.makedirs(DIR_PROCESSED, exist_ok=True)

linhas = []
for chave, cfg in SERIES_CONFIG.items():
    for ano, valor in zip(cfg["anos"], cfg["valores"]):
        linhas.append({
            "municipio":      MUNICIPIO,
            "indicador_id":   chave,
            "indicador_nome": cfg["titulo"],
            "df_origem":      cfg["df_key"],
            "grupo_censo":    cfg["grupo"],
            "n_censos":       cfg["n_censos"],
            "ano":            ano,
            "valor":          valor,
            "unidade_medida": cfg["ylabel"],
        })

df_export  = pd.DataFrame(linhas)
nome_base  = MUNICIPIO.lower().replace(' ', '_')
path_csv   = f"{DIR_PROCESSED}/indicadores_{nome_base}_tratados.csv"
path_parq  = f"{DIR_PROCESSED}/indicadores_{nome_base}_tratados.parquet"

df_export.to_csv(path_csv, index=False, encoding='utf-8')
df_export.to_parquet(path_parq, index=False)

print(f"✅ Exportado:\n  {path_csv}\n  {path_parq}")
display(df_export.head(10))


# ═══════════════════════════════════════════════════════════════════════════
# 7) DEPLOY NO GITHUB
# ═══════════════════════════════════════════════════════════════════════════

def publicar_no_github():
    from google.colab import userdata
    token     = userdata.get('GITHUB_TOKEN')
    auth_url  = f"https://{GITHUB_USUARIO}:{token}@github.com/{GITHUB_USUARIO}/{GITHUB_REPO}.git"
    repo_path = f'/content/{GITHUB_REPO}'

    print("🧹 Limpando cache local...")
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

    print(f"📥 Clonando branch '{GITHUB_BRANCH}'...")
    os.chdir('/content')
    os.system(f'git clone -b {GITHUB_BRANCH} {auth_url} {repo_path}')

    os.chdir(repo_path)
    os.system(f'git config user.email "{GITHUB_EMAIL}"')
    os.system(f'git config user.name "{GITHUB_USUARIO}"')

    destino = os.path.join(repo_path, GITHUB_PASTA)
    os.makedirs(destino, exist_ok=True)

    print(f"📁 Copiando dados para '{GITHUB_PASTA}'...")
    for arq in glob.glob(f'{DIR_PROCESSED}/*'):
        shutil.copy(arq, destino)

    for nb in glob.glob('/content/*.ipynb'):
        shutil.copy(nb, repo_path)

    print("🚀 Commit e push...")
    os.chdir(repo_path)
    os.system('git add .')
    os.system(f'git commit -m "feat: dados {MUNICIPIO} — indicadores_{nome_base}_tratados"')
    os.system(f'git push origin {GITHUB_BRANCH}')
    print(f"\n✅ Deploy concluído → {GITHUB_REPO}/{GITHUB_PASTA}")


publicar_no_github()
