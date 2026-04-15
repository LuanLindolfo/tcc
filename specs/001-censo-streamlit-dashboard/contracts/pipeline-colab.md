# Contrato de Pipeline: Google Colab → GitHub

**Projeto**: Sistema de Inteligência Territorial de Castanhal  
**Data**: 2026-03-16

> **Constituição v1.1.0 Princípio V**: O notebook **fonte de verdade** está em
> https://colab.research.google.com/drive/1DI1Xzzeo1JjgIgJQQr80zfOTLICUTG3p
> O arquivo `notebooks/censo_castanhal_pipeline.ipynb` no repo é espelho.

---

## Estrutura do Notebook Colab

O notebook principal (`censo_castanhal_pipeline.ipynb`) é organizado em seções executáveis sequencialmente.

### Seções obrigatórias (células em ordem)

```
[Célula 1] Setup & Autenticação
[Célula 2] Ingestão: Google Drive → DataFrames
[Célula 3] Limpeza e Normalização
[Célula 4] Engenharia de Features (variáveis compostas)
[Célula 5] Análise Exploratória (EDA — opcional, visualização local)
[Célula 6] Treinamento: Modelo de Classificação (IVS)
[Célula 7] Treinamento: Modelo de Regressão (IAH)
[Célula 8] Treinamento: Clustering (Ocupação)
[Célula 9] Exportação de Artefatos (.parquet + .joblib + .json)
[Célula 10] Push para GitHub
```

---

## Contrato de Célula 1: Setup & Autenticação

**Pré-condições**:
- `GITHUB_TOKEN` configurado em Colab Secrets
- Google Drive montado (via `drive.mount`)

**Assinaturas obrigatórias**:
```python
from google.colab import drive, userdata
drive.mount('/content/drive')

GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
GITHUB_REPO = "SEU_USUARIO/SEU_REPO"
GITHUB_BRANCH = "main"
DRIVE_BASE_PATH = "/content/drive/MyDrive/TCC_Castanhal/dados/"
```

---

## Contrato de Célula 2: Ingestão

**Entradas**: Arquivos `.xlsx` no Google Drive  
**Saídas**: `dict[str, pd.DataFrame]` com chaves: `"demografico"`, `"domicilios"`, `"educacao"`, `"trabalho_renda"`

**Assinaturas obrigatórias**:
```python
def carregar_xlsx(caminho: str) -> pd.DataFrame:
    """Carrega arquivo XLSX e retorna DataFrame."""

def ingerir_dados(base_path: str) -> dict[str, pd.DataFrame]:
    """
    Retorna dicionário com todos os DataFrames brutos.
    Raises: FileNotFoundError se arquivo não encontrado.
    """
```

---

## Contrato de Célula 3: Limpeza

**Entradas**: `dict[str, pd.DataFrame]` brutos  
**Saídas**: `dict[str, pd.DataFrame]` limpos

**Tabelas IBGE com metadados**: Para arquivos XLSX com metadados no topo (ex: `taxa_atividade.xlsx`, `profissões.xlsx`, `distribuição de renda.xlsx`), usar as funções documentadas em [ibge-table-cleaning.md](./ibge-table-cleaning.md): `limpar_taxa_atividade`, `limpar_taxa_atividade_percentual`, `limpar_profissoes`, `limpar_distribuicao_renda`, `extrair_esc`. Padrão: `header=None` → localizar dados por texto → extrair colunas → remover vazios/Total → `reset_index(drop=True)`.

**Operações obrigatórias**:
```python
def limpar_dataframe(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    1. Remove duplicatas por setor_id
    2. Converte tipos (str→float para percentuais)
    3. Padroniza nomes de colunas (snake_case, sem acentos)
    4. Imputa NaN com mediana para colunas numéricas
    5. Valida ranges (percentuais 0-100, renda ≥ 0)
    """
```

**Invariante de limpeza**: Após limpeza, nenhuma coluna numérica deve conter `NaN`.

---

## Contrato de Célula 4: Engenharia de Features

**Entradas**: DataFrames limpos  
**Saídas**: `features_compostas` DataFrame

**Features a calcular obrigatoriamente**:
```python
def calcular_iah(df_domicilios: pd.DataFrame) -> pd.Series:
    """IAH = média de (pct_agua + pct_esgoto + pct_lixo + pct_energia) / 100"""

def calcular_ide(df_demografico: pd.DataFrame) -> pd.Series:
    """IDE = (pop_0_14 + pop_65_mais) / pop_15_64"""

def calcular_ivs(df_features: pd.DataFrame, n_clusters: int = 3) -> pd.Series:
    """
    Aplica KMeans sobre (renda_norm, analfabetismo_norm, iah_inv).
    Retorna labels: 'baixa', 'media', 'alta'.
    """

def criar_features_compostas(*dataframes) -> pd.DataFrame:
    """Faz join de todos os DataFrames por setor_id e adiciona features compostas."""
```

---

## Contrato de Células 6–8: Treinamento ML

**Assinaturas obrigatórias**:

```python
def treinar_classificacao(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[RandomForestClassifier, dict]:
    """
    Treina RandomForest para IVS.
    Retorna (modelo_treinado, metricas_dict).
    metricas_dict inclui: acuracia, f1_macro, matriz_confusao, feature_importances.
    """

def treinar_regressao(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[XGBRegressor, dict]:
    """
    Treina XGBoost para IAH.
    Retorna (modelo_treinado, metricas_dict).
    metricas_dict inclui: r2, rmse, mae.
    """

def treinar_clustering(
    X: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42
) -> tuple[KMeans, np.ndarray]:
    """
    Treina KMeans para perfil de ocupação.
    Retorna (modelo_treinado, labels_array).
    """
```

---

## Contrato de Célula 9: Exportação de Artefatos

**Saídas obrigatórias no sistema de arquivos local** (antes do push):

```
/content/
├── data/
│   ├── processed/
│   │   ├── demografico.parquet
│   │   ├── domicilios.parquet
│   │   ├── educacao.parquet
│   │   ├── trabalho_renda.parquet
│   │   └── features_compostas.parquet
│   └── results/
│       ├── ml_classificacao_metricas.json
│       ├── ml_regressao_metricas.json
│       ├── ml_clustering_labels.parquet
│       └── politicas_recomendacoes.json
└── models/
    ├── random_forest_ivs.joblib
    ├── xgboost_iah.joblib
    └── kmeans_ocupacao.joblib
```

**Assinaturas obrigatórias**:
```python
def exportar_artefatos(
    dataframes: dict[str, pd.DataFrame],
    modelos: dict[str, Any],
    metricas: dict[str, dict],
    output_dir: str = "/content"
) -> None:
    """Salva todos os artefatos nos diretórios corretos."""
```

---

## Contrato de Célula 10: Push GitHub

**Pré-condições**: `GITHUB_TOKEN` disponível, artefatos exportados

```python
def push_para_github(
    token: str,
    repo: str,
    branch: str,
    commit_message: str = "chore: update processed data and models"
) -> None:
    """
    Configura git, adiciona artefatos e faz push para GitHub.
    Raises: subprocess.CalledProcessError se push falhar.
    """
```

**Arquivos enviados ao GitHub** (nunca enviar dados brutos):
- `data/processed/*.parquet` ✅
- `data/results/*.json` ✅  
- `data/results/*.parquet` ✅
- `models/*.joblib` ✅
- `data/raw/*.xlsx` ❌ (bloqueado por .gitignore)
