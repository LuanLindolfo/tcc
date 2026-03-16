# Research: Sistema de Inteligência Territorial de Castanhal

**Branch**: `001-censo-streamlit-dashboard` | **Date**: 2026-03-16  
**Phase**: 0 — Resolução de incógnitas técnicas

---

## 1. Integração Colab → GitHub (push seguro de artefatos)

**Decisão**: Usar `git` via subprocess no Colab + GitHub Personal Access Token (PAT) armazenado em `userdata` do Colab Secrets (não exposto no código).

**Rationale**: O Colab Secrets (`google.colab.userdata`) funciona como cofre isolado por usuário — o token nunca aparece em outputs ou histórico de células. O push via HTTPS autenticado com PAT é o método mais simples e robusto sem precisar de chaves SSH.

**Padrão de implementação**:
```python
from google.colab import userdata
import subprocess

token = userdata.get('GITHUB_TOKEN')
repo_url = f"https://{token}@github.com/SEU_USUARIO/SEU_REPO.git"
subprocess.run(["git", "remote", "set-url", "origin", repo_url])
subprocess.run(["git", "add", "data/processed/", "models/"])
subprocess.run(["git", "commit", "-m", "chore: update processed data and models"])
subprocess.run(["git", "push", "origin", "main"])
```

**Alternativas consideradas**:
- GitHub Actions (descartado — pipeline é manual, não automatizado)
- `PyGithub` library (mais verboso sem vantagem clara)
- Chaves SSH no Colab (requer configuração manual a cada sessão)

---

## 2. Streamlit lendo artefatos do GitHub (sem autenticação)

**Decisão**: Usar URLs raw do GitHub para leitura direta de `.parquet` e `.json` com `pandas` e `requests`. Para repositório público, nenhum token é necessário no Streamlit.

**Padrão de implementação**:
```python
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/SEU_USUARIO/SEU_REPO/main"

@st.cache_data(ttl=3600)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(f"{BASE_URL}/{path}")

df_demografico = load_data("data/processed/demografico.parquet")
```

**Benefícios do `.parquet`**:
- 3–10x menor que CSV para dados tabulares
- Preserva tipos de dados (int, float, categorical)
- Leitura colunar (mais rápida para dashboards com filtros)

**Alternativas consideradas**:
- `.csv` (descartado — maior, sem tipos nativos)
- `.pickle` (descartado — inseguro para dados públicos, versão-dependente)
- `.joblib` para modelos (mantido — padrão scikit-learn)

---

## 3. Assistente de IA com Gemini (padrão RAG simplificado)

**Decisão**: Usar **context injection** (RAG simplificado) em vez de vector stores completos. Os dados processados são compactos o suficiente para caber no contexto do Gemini (128k tokens), tornando embeddings desnecessários para o TCC.

**Arquitetura**:
```
[Pergunta do usuário]
       ↓
[Gerar resumo estatístico dos dados relevantes como texto]
       ↓
[Injetar no prompt do Gemini junto com a pergunta]
       ↓
[Gemini retorna resposta contextualizada]
```

**Padrão de implementação**:
```python
import google.generativeai as genai

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

def gerar_contexto_dados(df: pd.DataFrame, tema: str) -> str:
    stats = df.describe().to_string()
    return f"Dados do Censo 2022 de Castanhal ({tema}):\n{stats}"

def consultar_ia(pergunta: str, contexto: str) -> str:
    prompt = f"""Você é um especialista em análise de dados censitários e políticas públicas.
    
Contexto dos dados:
{contexto}

Pergunta: {pergunta}

Responda de forma clara e objetiva, citando os dados quando relevante."""
    response = model.generate_content(prompt)
    return response.text
```

**Gemini API Key**: Armazenada em `st.secrets["GEMINI_API_KEY"]` no Streamlit Community Cloud (nunca no código).

**Alternativas consideradas**:
- LangChain + ChromaDB (descartado — overhead excessivo para TCC)
- OpenAI (descartado — usuário escolheu Gemini)
- Fine-tuning (descartado — dados insuficientes e custo alto)

---

## 4. Algoritmos de Machine Learning para dados censitários

### 4A. Classificação — Vulnerabilidade Socioeconômica

**Decisão**: **Random Forest Classifier** como modelo principal + **Gradient Boosting** para comparação.

**Rationale**:
- Handles mixed feature types (categorical + numerical) nativamente
- Feature importance interpretável (essencial para TCC e políticas públicas)
- Robusto a outliers e dados faltantes (com imputação simples)
- Sem necessidade de normalização extensiva

**Variável-alvo**: Índice composto criado via K-Means (3 clusters) sobre:
- Renda domiciliar per capita (normalizada)
- Taxa de analfabetismo
- % domicílios sem saneamento básico
- % domicílios com paredes inadequadas (taipa, madeira)

**Features de entrada**: Todas as variáveis de demografia, educação, domicílios e renda

**Métricas de avaliação**: Acurácia, F1-score (macro), matriz de confusão

### 4B. Regressão — Infraestrutura Urbana

**Decisão**: **Ridge Regression** como baseline + **XGBoost Regressor** como modelo final.

**Rationale**:
- Ridge: interpretável, bom baseline, indica correlações lineares
- XGBoost: captura não-linearidades, excelente performance em dados tabulares, robusto

**Variável-alvo**: Índice de Adequação Habitacional (IAH) composto por:
- % com água encanada
- % com esgoto sanitário  
- % com coleta de lixo
- % com energia elétrica

**Features de entrada**: Variáveis demográficas + localização (setor censitário) + renda

**Métricas de avaliação**: R², RMSE, MAE

### 4C. Clustering — Perfil Socioeconômico por Ocupação

**Decisão**: **K-Means** (k=3 ou 4 clusters) para segmentar setores censitários.

**Rationale**: Simples, interpretável, adequado para apresentação acadêmica, clusters facilmente nomeáveis.

**Features**: Distribuição PEA por setor econômico (primário/secundário/terciário) + renda média + escolaridade

---

## 5. Engenharia de Features (variáveis compostas)

| Feature Composta | Fórmula / Lógica | Uso |
|-----------------|-----------------|-----|
| **IAH** (Índice de Adequação Habitacional) | Média ponderada de % água + % esgoto + % lixo + % energia | Alvo de regressão + feature de classificação |
| **IDE** (Indicador de Dependência Econômica) | (Pop. 0-14 + Pop. 65+) / PEA | Feature demográfica |
| **IVS** (Índice de Vulnerabilidade Socioeconômica) | Cluster label de K-Means sobre renda + educação + habitação | Alvo de classificação |
| **Zona de Envelhecimento** | Índice de envelhecimento > limiar (a definir por quartil) | Feature binária para análise demográfica |

---

## 6. Estrutura de Artefatos no GitHub

```
data/
├── raw/              # NÃO commitado (.gitignore) — fica apenas no Drive
├── processed/
│   ├── demografico.parquet
│   ├── domicilios.parquet
│   ├── educacao.parquet
│   ├── trabalho_renda.parquet
│   └── features_compostas.parquet
└── results/
    ├── ml_classificacao_metricas.json
    ├── ml_regressao_metricas.json
    ├── ml_clustering_labels.parquet
    └── politicas_recomendacoes.json

models/
├── random_forest_ivs.joblib
├── xgboost_iah.joblib
└── kmeans_ocupacao.joblib
```

---

## 7. Bibliotecas e Versões Recomendadas

| Categoria | Biblioteca | Versão mínima | Uso |
|-----------|-----------|--------------|-----|
| Streamlit | `streamlit` | 1.32+ | App principal |
| Dados | `pandas` | 2.0+ | Manipulação DataFrames |
| Dados | `pyarrow` | 14.0+ | Leitura/escrita .parquet |
| ML | `scikit-learn` | 1.4+ | RF, Ridge, KMeans |
| ML | `xgboost` | 2.0+ | XGBoost Regressor |
| ML | `joblib` | 1.3+ | Serialização de modelos |
| Viz | `plotly` | 5.18+ | Gráficos interativos |
| IA | `google-generativeai` | 0.5+ | Gemini API |
| Segurança | `python-dotenv` | 1.0+ | Variáveis de ambiente local |

---

## 8. Segurança e Proteção de Dados

**Dados brutos (XLSX)**: Ficam **apenas no Google Drive** — nunca commitados no GitHub.

**Credenciais**:
- `GITHUB_TOKEN`: Colab Secrets (`userdata.get`)
- `GEMINI_API_KEY`: Streamlit Secrets (`st.secrets`) + `.env` local

**`.gitignore` obrigatório**:
```gitignore
data/raw/
*.xlsx
*.env
google_auth.json
credentials.json
*.pkl
__pycache__/
.ipynb_checkpoints/
```

**Dados públicos no GitHub**: Apenas dados agregados por setor censitário (não há dados individuais no Censo IBGE — dados já são anônimos por design).

---

## Incógnitas resolvidas

| Incógnita original | Decisão tomada |
|-------------------|---------------|
| Provedor de LLM | Google Gemini (`gemini-1.5-flash`) |
| Alvos de ML | Classificação: IVS (vulnerabilidade); Regressão: IAH (infraestrutura) |
| Deploy Streamlit | Streamlit Community Cloud + GitHub público |
| Pipeline trigger | Manual (execução do notebook Colab) |
| Autenticação | Pública, sem login |
| Formato de artefatos | `.parquet` (dados) + `.joblib` (modelos) + `.json` (métricas) |
| Push Colab→GitHub | PAT via Colab Secrets + subprocess git |
| RAG vs injeção de contexto | Context injection (suficiente para volume de dados do TCC) |
