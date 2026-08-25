# -*- coding: utf-8 -*-
"""
TCC — Projeções Censitárias com Redes Neurais (versão CORINGA v2)
Edite apenas o bloco CONFIGURAÇÃO para trocar de município.

Novidades desta versão em relação à anterior:
- Não usa mais 'relu'/'lbfgs' como padrão fixo. Para cada indicador com
  3 ou mais censos válidos, o script testa as 12 combinações de
  ativação × solver (a mesma matriz de `explorar_combinacoes_rna`) e
  escolhe automaticamente a de MENOR erro via validação leave-one-out
  (LOOCV) — ou seja, a "melhor escolha dos testes" fica automática.
- Indicadores com apenas 2 censos (Grupo 3) não têm como ser validados
  por LOOCV (não sobra ponto de teste), então usam o padrão 'relu'/'lbfgs'.
- Ao final, gera o GRÁFICO de projeção 2030/2040 de TODO indicador de
  TODO grupo (1, 2 e 3), automaticamente, mais uma tabela-resumo
  consolidada com todas as previsões.
- `OVERRIDES` continua funcionando: se você já sabe a melhor combinação
  de um indicador específico, pode travar manualmente e pular a busca.
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

MUNICIPIO = "Belém"

# Pasta raiz no Google Drive
BASE_DRIVE = "/content/drive/MyDrive/dados_ibge/Dados comparativos 90, 00, 10 e 22 - Belém"

# Caminhos RELATIVOS ao BASE_DRIVE (sem repetir o BASE_DRIVE aqui)
ARQUIVOS = {
    "composicaofamiliar":    "Composição familiar/composicao_familiar_censo.csv",
    "deslocamento":          "Deslocamento para trabalho e estudo/mobilidade_e_transporte_censo.csv",
    "domicilios":            "Domicílios/domicilios_censo.csv",
    "educacao":              "Educação/educacao_censo.csv",
    "entornodomicilios":     "Entorno dos domicílios/entorno_de_domicilios_censo.csv",
    "familiasenupcialidade": "Famílias e Nupcialidade/familias_e_nupcialidade_censo.csv",
    "favelasecomunidade":    "Favelas e comunidades urbanas/favelas_censo.csv",
    "indigenas":             "Indígenas/indigenas_censo.csv",
    "populacao":             "População/populacao_censo.csv",
    "quilombolas":           "Quilombolas/quilombola_censo.csv",
    "trabalhoerendimento":   "Trabalho e rendimento/trabalho_e_rendimento_censo.csv",
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
# Trava manualmente ativação/solver de um indicador específico, pulando a
# busca automática por LOOCV para ele. A chave é "<df_key>__<slug_do_indicador>"
# (o próprio script imprime a chave de cada indicador ao selecionar o modelo).
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

# ── Seleção automática do melhor modelo (ativação × solver) ────────────────
SELECAO_AUTOMATICA = True   # False = todo indicador usa 'relu'/'lbfgs' direto
LOOCV_MAX_ITER     = 3000   # max_iter usado durante a busca (menor = mais rápido)
FIT_FINAL_MAX_ITER = 5000   # max_iter usado no treino final (gráfico/previsão)

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
    raiz = os.path.dirname(BASE_DRIVE)
    print(f"🔎 Verificando: {raiz}")
    if not os.path.isdir(raiz):
        print(f"  ❌ Essa pasta-raiz nem existe. Conteúdo do diretório pai:")
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
    Tenta ler o CSV de forma tolerante: vírgula/ponto-e-vírgula,
    utf-8/latin-1, detecção automática de separador e, por fim,
    tolerância a linhas malformadas (on_bad_lines='skip').
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
            if df.shape[1] > 1:
                return df
        except Exception:
            continue

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
# 2) FUNÇÕES AUXILIARES — PARSING E TREINO DE RNA
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


def _treinar_mlp(anos, valores, ativacao, solver, anos_alvo,
                 max_iter=FIT_FINAL_MAX_ITER, **kwargs):
    """Treina um MLPRegressor padronizado e retorna a previsão para anos_alvo."""
    X = np.array(anos, dtype=float).reshape(-1, 1)
    y = np.array(valores, dtype=float).reshape(-1, 1)
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
            np.array(anos_alvo, dtype=float).reshape(-1, 1)
        )).reshape(-1, 1)
    )


def _loocv_mae(anos, valores, ativacao, solver, max_iter=LOOCV_MAX_ITER) -> float:
    """
    Erro médio absoluto (MAE) via validação leave-one-out: para cada ponto,
    treina com os demais e mede o erro na previsão do ponto deixado de fora.
    Quanto menor, melhor a combinação ativação/solver generaliza.
    """
    anos    = list(anos)
    valores = list(valores)
    n = len(anos)
    if n < 3:
        return np.inf

    erros = []
    for i in range(n):
        anos_treino = anos[:i] + anos[i + 1:]
        val_treino  = valores[:i] + valores[i + 1:]
        try:
            pred = _treinar_mlp(anos_treino, val_treino, ativacao, solver,
                                [anos[i]], max_iter=max_iter)
            erros.append(abs(pred[0][0] - valores[i]))
        except Exception:
            return np.inf
    return float(np.mean(erros))


_COMBINACOES_RNA = list(itertools.product(
    ['identity', 'logistic', 'tanh', 'relu'],
    ['lbfgs', 'sgd', 'adam'],
))


def selecionar_melhor_combo(anos, valores):
    """
    Testa as 12 combinações de ativação × solver e retorna a de menor
    erro LOOCV: (ativacao, solver, loocv_mae).
    Com menos de 3 pontos não há como validar (sobra 0 ou 1 ponto de
    teste) — nesse caso retorna o padrão 'relu'/'lbfgs' sem buscar.
    """
    if len(anos) < 3 or not SELECAO_AUTOMATICA:
        return "relu", "lbfgs", None

    melhor, melhor_erro = None, np.inf
    for ativ, solver in _COMBINACOES_RNA:
        erro = _loocv_mae(anos, valores, ativ, solver)
        if erro < melhor_erro:
            melhor_erro, melhor = erro, (ativ, solver)

    if melhor is None or not np.isfinite(melhor_erro):
        return "relu", "lbfgs", None
    return melhor[0], melhor[1], melhor_erro


# ═══════════════════════════════════════════════════════════════════════════
# 3) DESCOBERTA, AGRUPAMENTO E SELEÇÃO AUTOMÁTICA DE MODELO POR INDICADOR
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
           0 ou 1 censo     -> descartado
      5) para indicadores com 3+ censos, escolhe automaticamente a melhor
         combinação ativação/solver via LOOCV (`selecionar_melhor_combo`).
    """
    config = {}
    chaves_usadas = set()
    contagem_grupo = {"Grupo 1 — 4 censos": 0, "Grupo 2 — 3 censos": 0,
                       "Grupo 3 — 2 censos": 0, "descartado (≤1 censo)": 0}

    # Primeiro: monta a lista de candidatos (sem treinar nada ainda)
    candidatos = []
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

            candidatos.append(dict(
                chave=chave, df_key=df_key, titulo=indicador_txt,
                anos=anos, valores=valores, n_censos=n_censos, grupo=grupo,
            ))

    print("\n📊 Resumo do agrupamento automático:")
    for grupo, qtd in contagem_grupo.items():
        print(f"  {grupo}: {qtd} indicador(es)")

    n_com_busca = sum(1 for c in candidatos if c["n_censos"] >= 3)
    print(f"\n🧠 Selecionando a melhor combinação ativação/solver via LOOCV "
          f"para {n_com_busca} indicador(es) com 3+ censos "
          f"({len(_COMBINACOES_RNA)} combinações testadas cada). "
          f"Isso pode levar alguns minutos...\n")

    for i, cand in enumerate(candidatos, start=1):
        chave, anos, valores = cand["chave"], cand["anos"], cand["valores"]
        ov = OVERRIDES.get(chave, {})

        if "ativacao" in ov and "solver" in ov:
            ativ, solver, loocv_mae, auto = ov["ativacao"], ov["solver"], None, False
        else:
            ativ, solver, loocv_mae = selecionar_melhor_combo(anos, valores)
            auto = cand["n_censos"] >= 3
            if "ativacao" in ov:
                ativ = ov["ativacao"]
            if "solver" in ov:
                solver = ov["solver"]

        if auto and loocv_mae is not None:
            print(f"  [{i}/{len(candidatos)}] {chave} → '{ativ}'/'{solver}' "
                  f"(LOOCV MAE={loocv_mae:,.3f})")

        config[chave] = {
            "titulo":           cand["titulo"],
            "df_key":           cand["df_key"],
            "anos":             anos,
            "valores":          valores,
            "n_censos":         cand["n_censos"],
            "grupo":            cand["grupo"],
            "ylabel":           ov.get("ylabel", "Valor"),
            "ativacao":         ativ,
            "solver":           solver,
            "auto_selecionado": auto,
            "loocv_mae":        loocv_mae,
            "format_str":       ov.get("format_str", "{:.2f}"),
            "y_lim":            ov.get("y_lim"),
            "future_years":     ov.get("future_years", ANOS_FUTUROS_PADRAO),
            "curve_range":      ov.get("curve_range", (min(anos) - 5, max(anos) + 16)),
        }

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


# Exibição dos indicadores agrupados
for grupo_nome in ["Grupo 1 — 4 censos", "Grupo 2 — 3 censos", "Grupo 3 — 2 censos"]:
    _secao(f"{MUNICIPIO} — {grupo_nome}")
    linhas_grupo = [
        {"indicador": cfg["titulo"], "df": cfg["df_key"],
         "ativação": cfg["ativacao"], "solver": cfg["solver"],
         **{str(a): v for a, v in zip(cfg["anos"], cfg["valores"])}}
        for cfg in SERIES_CONFIG.values() if cfg["grupo"] == grupo_nome
    ]
    if linhas_grupo:
        display(pd.DataFrame(linhas_grupo))
    else:
        print("  (nenhum indicador nesse grupo)")


# ═══════════════════════════════════════════════════════════════════════════
# 4) MODELOS — GRID MANUAL (opcional, para inspeção visual pontual)
# ═══════════════════════════════════════════════════════════════════════════

def explorar_combinacoes_rna(chave: str, config=None):
    """Grid 4×3 (ativação × solver) para inspecionar visualmente um indicador."""
    if config is None:
        config = SERIES_CONFIG
    cfg = config[chave]
    anos, valores = cfg["anos"], cfg["valores"]

    if len(anos) < 2:
        print(f"⚠️  '{chave}' tem menos de 2 pontos — não é possível treinar.")
        return

    anos_curva = np.arange(*cfg["curve_range"]).reshape(-1, 1)
    anos_fut   = np.array(cfg["future_years"])

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(f"{cfg['titulo']} — {MUNICIPIO} ({cfg['grupo']})\nMatriz de RNAs",
                 fontsize=18, y=0.92)
    print(f"Treinando 12 RNAs para '{chave}'...")

    for ax, (ativ, solver) in zip(axes.flatten(), _COMBINACOES_RNA):
        tend = _treinar_mlp(anos, valores, ativ, solver, anos_curva.ravel())
        prev = _treinar_mlp(anos, valores, ativ, solver, anos_fut)
        ax.scatter(anos, valores, color='royalblue', s=100, zorder=5, label='Dados Reais')
        ax.plot(anos_curva, tend, color='indigo', linewidth=2.5)
        ax.scatter(anos_fut, prev, color='red', s=80, zorder=5)
        fmt = cfg["format_str"]
        marca = " ★" if (ativ, solver) == (cfg["ativacao"], cfg["solver"]) else ""
        ax.set_title(f"'{ativ}' | '{solver}'{marca}\nPrev {anos_fut[-1]}: {fmt.format(prev[-1][0])}",
                     fontsize=11)
        ax.set_ylabel(cfg["ylabel"])
        if cfg["y_lim"]:
            ax.set_ylim(*cfg["y_lim"])
        ax.grid(True, linestyle=':', alpha=0.7)

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# 5) GRÁFICO FINAL + PROJEÇÃO 2030/2040 — PARA TODO INDICADOR DE TODO GRUPO
# ═══════════════════════════════════════════════════════════════════════════

RESULTADOS_PROJECAO = []  # acumula todas as previsões geradas pelos plots


def plot_final_model(chave: str, config=None):
    """
    Gráfico final com a ativação/solver já escolhida (via LOOCV ou override)
    para o indicador, com a projeção para `future_years` (padrão 2030/2040).
    Cada previsão gerada é também registrada em RESULTADOS_PROJECAO.
    """
    if config is None:
        config = SERIES_CONFIG
    cfg = config[chave]
    anos, valores = cfg["anos"], cfg["valores"]
    ativ, solver  = cfg["ativacao"], cfg["solver"]
    fmt           = cfg["format_str"]

    if len(anos) < 2:
        print(f"⚠️  '{chave}' tem menos de 2 pontos — gráfico ignorado.")
        return

    X  = np.array(anos, dtype=float).reshape(-1, 1)
    y  = np.array(valores, dtype=float).reshape(-1, 1)
    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X)
    ys = sy.fit_transform(y)

    m = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS, activation=ativ,
        solver=solver, max_iter=FIT_FINAL_MAX_ITER, random_state=RANDOM_STATE,
    )
    m.fit(Xs, ys.ravel())

    Xc = np.arange(*cfg["curve_range"]).reshape(-1, 1)
    yc = sy.inverse_transform(m.predict(sx.transform(Xc)).reshape(-1, 1))
    Xf = np.array(cfg["future_years"]).reshape(-1, 1)
    yf = sy.inverse_transform(m.predict(sx.transform(Xf)).reshape(-1, 1))

    subtitulo = f"Ativação: '{ativ}' | Solver: '{solver}'"
    if cfg.get("auto_selecionado") and cfg.get("loocv_mae") is not None:
        subtitulo += f"  (auto — LOOCV MAE={cfg['loocv_mae']:,.2f})"

    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, color='blue', s=100, zorder=5, label='Dados Reais (IBGE)')
    plt.plot(Xc, yc, color='indigo', linewidth=2.5, label='Curva RNA')
    plt.scatter(Xf, yf, color='red', s=100, zorder=5, label='Projeção')
    plt.title(f"{cfg['titulo']} — {MUNICIPIO} ({cfg['grupo']})\n{subtitulo}",
              fontsize=12, pad=15)
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

    for ano, valor_prev in zip(cfg["future_years"], yf.ravel()):
        RESULTADOS_PROJECAO.append({
            "indicador_id":     chave,
            "indicador_nome":   cfg["titulo"],
            "df_origem":        cfg["df_key"],
            "grupo_censo":      cfg["grupo"],
            "ativacao":         ativ,
            "solver":           solver,
            "auto_selecionado": cfg.get("auto_selecionado", False),
            "loocv_mae":        cfg.get("loocv_mae"),
            "ano_previsto":     ano,
            "valor_previsto":   valor_prev,
        })


def gerar_todos_os_graficos_finais(config=None, grupo: str = None):
    """
    Gera o gráfico final (+ projeção 2030/2040) de todos os indicadores
    em SERIES_CONFIG. Se `grupo` for informado, filtra só esse grupo.
    """
    if config is None:
        config = SERIES_CONFIG
    for chave, cfg in config.items():
        if grupo and cfg["grupo"] != grupo:
            continue
        plot_final_model(chave, config)


# ── Gera o plot + projeção de TODO indicador, organizado por grupo ─────────
for grupo_nome in ["Grupo 1 — 4 censos", "Grupo 2 — 3 censos", "Grupo 3 — 2 censos"]:
    qtd = sum(1 for c in SERIES_CONFIG.values() if c["grupo"] == grupo_nome)
    _secao(f"{MUNICIPIO} — {grupo_nome} ({qtd} indicador(es)) — gráficos e projeção 2030/2040")
    if qtd == 0:
        print("  (nenhum indicador nesse grupo)")
        continue
    gerar_todos_os_graficos_finais(grupo=grupo_nome)


# ── Tabela-resumo consolidada de todas as projeções ─────────────────────────
_secao(f"{MUNICIPIO} — Tabela-resumo de projeções (2030/2040), todos os grupos")
df_projecoes = pd.DataFrame(RESULTADOS_PROJECAO)
if not df_projecoes.empty:
    df_projecoes = df_projecoes.sort_values(
        ["grupo_censo", "indicador_nome", "ano_previsto"]
    ).reset_index(drop=True)
    display(df_projecoes)
else:
    print("Nenhuma projeção gerada.")


# ═══════════════════════════════════════════════════════════════════════════
# 6) MODELOS ALTERNATIVOS DE POPULAÇÃO
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
    ativ, solver = _pop_cfg.get("ativacao", "relu"), _pop_cfg.get("solver", "lbfgs")
    tend = _treinar_mlp(ANOS_POP, POPULACAO, ativ, solver, CURVA_POP)
    prev = _treinar_mlp(ANOS_POP, POPULACAO, ativ, solver, _ANOS_FUT)
    print(f"--- RNA ({ativ}/{solver}) ---")
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
    ativ, solver = _pop_cfg.get("ativacao", "relu"), _pop_cfg.get("solver", "lbfgs")
    print(f"--- ENSEMBLE {n} RNAs ({ativ}/{solver}) ---  aguarde...")
    lp, lt = [], []
    for i in range(n):
        lp.append(_treinar_mlp(ANOS_POP, POPULACAO, ativ, solver, _ANOS_FUT,
                               max_iter=2000, alpha=0.5, random_state=i))
        lt.append(_treinar_mlp(ANOS_POP, POPULACAO, ativ, solver, CURVA_POP,
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
# 7) EXPORTAÇÃO CSV / PARQUET
#    - indicadores_<municipio>_tratados.*  → série histórica (dado real)
#    - projecoes_<municipio>_2030_2040.*   → previsões geradas nos plots
# ═══════════════════════════════════════════════════════════════════════════

os.makedirs(DIR_PROCESSED, exist_ok=True)

linhas = []
for chave, cfg in SERIES_CONFIG.items():
    for ano, valor in zip(cfg["anos"], cfg["valores"]):
        linhas.append({
            "municipio":        MUNICIPIO,
            "indicador_id":     chave,
            "indicador_nome":   cfg["titulo"],
            "df_origem":        cfg["df_key"],
            "grupo_censo":      cfg["grupo"],
            "n_censos":         cfg["n_censos"],
            "ativacao":         cfg["ativacao"],
            "solver":           cfg["solver"],
            "auto_selecionado": cfg.get("auto_selecionado", False),
            "loocv_mae":        cfg.get("loocv_mae"),
            "ano":              ano,
            "valor":            valor,
            "unidade_medida":   cfg["ylabel"],
        })

df_export  = pd.DataFrame(linhas)
nome_base  = MUNICIPIO.lower().replace(' ', '_')
path_csv   = f"{DIR_PROCESSED}/indicadores_{nome_base}_tratados.csv"
path_parq  = f"{DIR_PROCESSED}/indicadores_{nome_base}_tratados.parquet"

df_export.to_csv(path_csv, index=False, encoding='utf-8')
df_export.to_parquet(path_parq, index=False)

path_proj_csv  = f"{DIR_PROCESSED}/projecoes_{nome_base}_2030_2040.csv"
path_proj_parq = f"{DIR_PROCESSED}/projecoes_{nome_base}_2030_2040.parquet"
if not df_projecoes.empty:
    df_projecoes.insert(0, "municipio", MUNICIPIO)
    df_projecoes.to_csv(path_proj_csv, index=False, encoding='utf-8')
    df_projecoes.to_parquet(path_proj_parq, index=False)

print(f"✅ Exportado:\n  {path_csv}\n  {path_parq}\n  {path_proj_csv}\n  {path_proj_parq}")
display(df_export.head(10))


# ═══════════════════════════════════════════════════════════════════════════
# 8) DEPLOY NO GITHUB
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
    os.system(f'git commit -m "feat: dados {MUNICIPIO} — indicadores_{nome_base}_tratados + projecoes"')
    os.system(f'git push origin {GITHUB_BRANCH}')
    print(f"\n✅ Deploy concluído → {GITHUB_REPO}/{GITHUB_PASTA}")


publicar_no_github()
