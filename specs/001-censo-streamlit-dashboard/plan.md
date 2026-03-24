# Implementation Plan: Sistema de Inteligência Territorial de Castanhal

**Branch**: `001-censo-streamlit-dashboard` | **Date**: 2026-03-16  
**Spec**: [spec.md](./spec.md) | **Research**: [research.md](./research.md)

---

## Summary

Dashboard interativo em Streamlit para análise dos dados do Censo 2022 de Castanhal–PA, com modelos de ML (Random Forest para vulnerabilidade socioeconômica, XGBoost para infraestrutura urbana, K-Means para perfil de ocupação) e assistente conversacional powered by Google Gemini. O pipeline integra Google Colab (processamento manual) → GitHub (armazenamento de artefatos `.parquet`/`.joblib`) → Streamlit Community Cloud (visualização pública).

---

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: streamlit 1.36+ (usa `st.navigation`), pandas 2.0+, plotly 5.18+, scikit-learn 1.4+, xgboost 2.0+, google-generativeai 0.5+, pyarrow 14.0+, joblib 1.3+  
**Storage**: GitHub público — XLSX do IBGE em `censo_castanhal/censo_castanhal/` (dados públicos), artefatos processados em `data/processed/` (`.parquet`), `models/` (`.joblib`) e `data/results/` (`.json`)  
**Testing**: Testes manuais via Streamlit + validação de métricas ML no Colab  
**Target Platform**: Streamlit Community Cloud (Linux, Python 3.11)  
**Performance Goals**: Dashboard carrega em < 5s (dados cacheados via `@st.cache_data`); resposta do Gemini em < 10s  
**Constraints**: App público sem autenticação; dados brutos nunca no GitHub; chaves de API via `st.secrets`; pipeline disparo manual  
**Scale/Scope**: ~1 usuário principal + banca avaliadora; dados de ~1 município (Castanhal); volume estimado < 50MB de artefatos

---

## Constitution Check

*GATE: Verificado contra Constituição v1.1.0 (ratificada 2026-03-16).*

| Princípio Constitucional | Status | Evidência no Plano |
|--------------------------|--------|--------------------|
| **I. LGPD & Anonimização** | ✅ PASS | XLSX do IBGE são dados públicos, commitados em `censo_castanhal/censo_castanhal/`; GitHub recebe também `.parquet` agregados por setor em `data/processed/`; nenhum dado individual identificável em nenhum dos arquivos |
| **II. IA Ética** | ✅ PASS | Feature importances exportadas ao GitHub; métricas exibidas junto com resultados; Gemini como "apoio à decisão" |
| **III. Zero-Exposure de Credenciais** | ✅ PASS | `GITHUB_TOKEN` via Colab Secrets; `GEMINI_API_KEY` via `st.secrets`; `.gitignore` bloqueia `.env` e `secrets.toml` |
| **IV. Rigor Acadêmico + .parquet** | ✅ PASS | Todos os artefatos em `.parquet`/`.joblib`; `random_state=42`; `@st.cache_data`; `GITHUB_RAW_BASE` via secret |
| **V. Notebook Colab como Fonte de Verdade** | ✅ PASS | Notebook canônico em Colab Drive; `notebooks/censo_castanhal_pipeline.ipynb` é espelho; alterações fluem Colab → GitHub |
| **Índice Gini (diferido)** | ⏳ DEFERIDO | Campo `indice_gini` reservado no schema; cálculo e exibição aguardam confirmação de microdados |

**Resultado**: Nenhuma violação. Todos os 5 princípios satisfeitos. Prosseguir para implementação.

---

## Project Structure

### Documentação (esta feature)

```text
specs/001-censo-streamlit-dashboard/
├── plan.md              ← Este arquivo
├── spec.md              ← Especificação de requisitos
├── research.md          ← Decisões técnicas e pesquisa
├── data-model.md        ← Modelo de dados e entidades
├── quickstart.md        ← Guia de setup e configuração
├── contracts/
│   ├── streamlit-pages.md   ← Contrato de interface das páginas
│   ├── pipeline-colab.md    ← Contrato do pipeline Colab→GitHub
│   └── ibge-table-cleaning.md ← Funções de limpeza de tabelas IBGE (metadados no topo)
└── tasks.md             ← (gerado por /speckit.tasks)
```

### Código-Fonte (raiz do repositório)

```text
tcc-castanhal/
├── app.py                          ← Único entry point: st.navigation + funções render_* por seção
├── requirements.txt                ← Dependências Python
├── .gitignore                      ← Protege dados brutos e credenciais
├── .streamlit/
│   └── secrets.toml               ← NÃO commitar (Streamlit Cloud lê automaticamente)
│
├── utils/
│   ├── data_loader.py             ← Carga de .parquet do GitHub (cached)
│   ├── ml_utils.py                ← Carga de modelos .joblib
│   └── gemini_utils.py            ← Integração Google Gemini API
│
├── data/
│   ├── raw/                       ← ❌ .gitignore (apenas no Drive)
│   ├── processed/                 ← ✅ *.parquet (gerados pelo Colab)
│   └── results/                   ← ✅ *.json + *.parquet (métricas ML)
│
├── models/                        ← ✅ *.joblib (modelos treinados)
│
└── notebooks/
    └── censo_castanhal_pipeline.ipynb  ← Notebook principal (Colab)

Raiz (análise TCC):
├── tcc_tabelas_merge.ipynb  ← Tabelas consolidadas + seção 8: df_geral_municipal (junção única; convenção de nomes: specs/001-censo-streamlit-dashboard/data-model.md#df_geral_municipal)
└── tcc_censo_2022.ipynb     ← Análises exploratórias auxiliares
```

**Decisão de estrutura**: Projeto único (single-project). O Streamlit é o frontend e o Colab é o backend de processamento. Não há servidor separado — o Streamlit lê artefatos estáticos do GitHub. **Interface**: todas as seções vivem em `app.py` (`st.navigation`); não há pasta `pages/` (o uso de `st.navigation` faz o Streamlit ignorar `pages/` mesmo que exista).

---

## Complexity Tracking

Nenhuma violação de princípios detectada. Tabela não aplicável.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE DE DADOS                         │
│                                                             │
│  Google Drive          Google Colab           GitHub        │
│  ┌──────────┐    lê    ┌──────────────┐  push ┌─────────┐  │
│  │ *.xlsx   │ ──────→  │  notebook    │ ────→ │.parquet │  │
│  │ (brutos) │          │  limpeza     │       │.joblib  │  │
│  └──────────┘          │  ML treino   │       │.json    │  │
│                        └──────────────┘       └────┬────┘  │
│                         (manual)                   │ raw   │
└─────────────────────────────────────────────────────│───────┘
                                                      │ GitHub raw URLs
┌─────────────────────────────────────────────────────│───────┐
│                    STREAMLIT APP                     │       │
│                                                      ↓       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  utils/data_loader.py  (@st.cache_data ttl=3600)    │    │
│  └────────────────┬──────────────────────────────────┘    │
│                   ↓                                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │  app.py — st.navigation (Início, Demografia,        │    │
│  │  Domicílios, Educação & Renda, Políticas, Assistente)│    │
│  └────────────────────────────────────────────────────┘    │
│                   ↑                                         │
│  ┌────────────────┴───────────────────────────────────┐    │
│  │  Google Gemini API (via st.secrets["GEMINI_API_KEY"])│   │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  Deploy: Streamlit Community Cloud (URL pública)            │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Fase A: Estrutura Base e Pipeline Colab (P1 — Fundação)

**Objetivo**: Pipeline funcional que produz artefatos consumíveis no Streamlit.

**Entregáveis**:
1. Repositório GitHub com estrutura de diretórios e `.gitignore`
2. `requirements.txt` com todas as dependências
3. `notebooks/censo_castanhal_pipeline.ipynb` com todas as 10 células implementadas
4. Artefatos gerados e visíveis no GitHub: `data/processed/*.parquet`, `models/*.joblib`, `data/results/*.json`

**Critério de conclusão**: Executar o notebook do início ao fim sem erros e ver os arquivos no GitHub.

**Dependências**: Nenhuma.

---

### Fase B: App Streamlit — Estrutura e Dados (P1 — Core)

**Objetivo**: App funcional com as 3 abas de visualização de dados base.

**Entregáveis**:
1. `app.py` com `st.set_page_config`, `st.navigation` e funções `render_*` por seção
2. `utils/data_loader.py` com todas as funções de carga (com cache)
3. Seções de dados em `app.py`: Demografia, Domicílios, Educação & Renda (equivalente ao que antes estava em arquivos separados em `pages/`)

**Critério de conclusão**: As três seções de exploração de dados carregam com dados reais do GitHub sem erro.

**Dependências**: Fase A concluída (artefatos no GitHub).

---

### Fase C: Resultados de ML (P2 — Diferencial)

**Objetivo**: Modelos treinados no Colab e artefatos no GitHub; contextualização nas políticas e no assistente.

**Entregáveis**:
1. `utils/ml_utils.py` com funções de carga e predição (consumo opcional pelo app)
2. Artefatos em `data/results/*.json`, `models/*.joblib` — utilizados pela seção **Políticas** (`load_features_compostas`, JSON de políticas) e pelo contexto do Gemini

**Estado atual (UI)**: não há seção dedicada exclusivamente a “Machine Learning” no Streamlit; métricas e gráficos de modelos não são exibidos em uma aba própria.

**Dependências**: Fase A + Fase B concluídas.

---

### Fase D: Assistente Gemini (P2 — IA Conversacional)

**Objetivo**: Chat contextualizado sobre os dados do censo.

**Entregáveis**:
1. `utils/gemini_utils.py` — funções de geração de contexto e consulta (modelo configurável, ex.: `gemini-2.5-flash`)
2. Seção **Assistente IA** em `app.py` (`render_assistente_ia`) — chat com histórico (`st.chat_input`, `st.session_state`)
3. Configuração de `st.secrets["GEMINI_API_KEY"]` documentada

**Critério de conclusão**: Resposta coerente a ao menos 3 perguntas diferentes sobre os dados.

**Dependências**: Fase B concluída (dados carregados para contexto).

---

### Fase E: Políticas Públicas (P3 — Aplicação)

**Objetivo**: Conectar análises ML a recomendações concretas de políticas municipais.

**Entregáveis**:
1. `data/results/politicas_recomendacoes.json` populado (no Colab)
2. Seção **Políticas** em `app.py` (`render_politicas`) — seletor de área, cards, tabela/gráfico de setores prioritários
3. Ao menos 3 políticas mapeadas: Habitação Popular, Combate ao Analfabetismo, Geração de Emprego

**Critério de conclusão**: Aba exibe 3+ políticas com setores prioritários identificados pelos modelos.

**Dependências**: Fase C concluída (resultados ML com setores identificados).

---

## Key Design Decisions

| Decisão | Escolha | Justificativa |
|---------|---------|--------------|
| Formato de artefatos | `.parquet` | Compacto, tipado, leitura colunar eficiente |
| Pipeline disparo | Manual (Colab) | Dados Censo não mudam; simplicidade > automação |
| Deploy | Streamlit Community Cloud | Gratuito, integra diretamente com GitHub |
| LLM | Google Gemini (`gemini-2.5-flash` no app) | API Google AI; modelo atualizado conforme disponibilidade |
| Padrão IA | Context injection (não RAG completo) | Volume de dados cabe no contexto; sem overhead de vector store |
| Modelos ML | RF + XGBoost + K-Means | Interpretáveis, robustos, padrão acadêmico |
| Segredos | Colab Secrets + Streamlit Secrets | Nunca expostos em código ou histórico |

---

## Risk Mitigation

| Risco | Probabilidade | Impacto | Mitigação |
|-------|--------------|---------|----------|
| Cota Gemini API esgotada | Média | Alto | Usar modelo flash atual (`gemini-2.5-flash` ou equivalente); implementar fallback de mensagem |
| GitHub raw URL lenta | Baixa | Médio | `@st.cache_data(ttl=3600)` evita re-downloads frequentes |
| Sessão Colab expira durante treinamento | Alta | Médio | Salvar checkpoints após cada modelo; re-executar apenas células necessárias |
| Dados XLSX com estrutura variada | Alta | Alto | Implementar limpeza defensiva com nomes de colunas flexíveis + log de colunas não encontradas |
| Push GitHub falha (token expirado) | Baixa | Médio | Verificar token antes do push; instruções de renovação no quickstart |
