# 🏙️ Sistema de Inteligência Territorial de Castanhal

Dashboard interativo para análise dos dados do **Censo IBGE** (2010–2022) do município de
Castanhal – PA, com modelos de Machine Learning e assistente de IA conversacional.

> **Projeto**: Trabalho de Conclusão de Curso (TCC)
> **Constituição**: v1.0.0 — LGPD, IA Ética, Zero-Exposure, Rigor Acadêmico

---

## 🏗️ Arquitetura do Pipeline

```
GitHub (censo_castanhal/censo_castanhal/*.xlsx — dados IBGE públicos)
      ↓  [notebooks/censo_castanhal_pipeline.ipynb — baixa via URL raw]
Google Colab (limpeza + feature engineering + treinamento ML)
      ↓  [push via PAT — Colab Secrets: GITHUB_TOKEN]
GitHub (data/processed/*.parquet + models/*.joblib + data/results/*.json)
      ↓  [GitHub raw URLs + @st.cache_data]
Streamlit Community Cloud (visualização pública)
```

**Repositório**: https://github.com/LuanLindolfo/tcc
**Dados XLSX**: https://github.com/LuanLindolfo/tcc/tree/main/censo_castanhal/censo_castanhal

---

## 📊 Funcionalidades

O app é um único `app.py` com navegação via `st.navigation` (sem pasta `pages/`). Seções:

| Seção | Conteúdo |
|--------|----------|
| 🏠 Início | Boas-vindas e visão geral do dashboard |
| 📊 Demografia | Pirâmide etária, distribuição étnico-racial, KPIs |
| 🏠 Domicílios | IAH, saneamento básico, tipos de domicílio |
| 📚 Educação & Renda | Escolaridade, distribuição de renda, PEA |
| 🏛️ Políticas Públicas | Políticas com setores prioritários (usa artefatos de ML do pipeline) |
| 💬 Assistente IA | Chat contextualizado com Google Gemini |

> **Machine Learning**: os modelos continuam sendo treinados no Colab e os artefatos ficam no GitHub (`models/`, `data/results/`); não há seção dedicada só a ML no Streamlit — os resultados entram no contexto da aba **Políticas** e do assistente.

---

## 🚀 Instalação Local

```bash
git clone https://github.com/LuanLindolfo/tcc.git
cd tcc

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt

# Criar arquivo de secrets local
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edite secrets.toml com suas chaves

streamlit run app.py
```

---

## 🔑 Configuração de Secrets

**Streamlit Community Cloud** (produção):
- `GEMINI_API_KEY` → chave do Google AI Studio (https://aistudio.google.com/app/apikey)
- `GITHUB_RAW_BASE` → `https://raw.githubusercontent.com/LuanLindolfo/tcc/001-censo-streamlit-dashboard`

**Google Colab** (pipeline):
- `GITHUB_TOKEN` → Personal Access Token com permissão `Contents: Write`

---

## 📓 Notebooks

| Notebook | Propósito |
|----------|-----------|
| `notebooks/censo_castanhal_pipeline.ipynb` | Pipeline Colab (fonte de verdade): limpeza + ML + push GitHub |
| `tcc_tabelas_merge.ipynb` | Tabelas consolidadas por tópico do documento TCC (ver DataFrames abaixo) |
| `tcc_censo_2022.ipynb` | Análises exploratórias e processamento auxiliar |

### DataFrames gerados pelo `tcc_tabelas_merge.ipynb`

**Período dos dados**: 2010–2022 (Censo IBGE). Dados anteriores a 2010 não são utilizados. **Usa Censo 2010**: apenas `Taxa_Crescimento_Anual` (comparação 2010→2022).

| DataFrame | O que é |
|-----------|---------|
| `df_demografia` | Indicadores populacionais: população total, densidade, urbanização, taxa de crescimento, índice de envelhecimento e razão de sexo do município |
| `df_domicilios` | Características habitacionais: tipos de domicílio (casa/apartamento), saneamento, material das paredes e condições de ocupação (próprio/alugado/cedido) |
| `df_educacao` | Escolaridade por sexo: distribuição de pessoas de 18+ anos por nível de instrução (sem instrução, fundamental, médio, superior) |
| `df_trabalho_renda` | Condição de atividade econômica: PEA ocupada, PEA desocupada e não economicamente ativas |
| `df_renda` | Renda domiciliar mensal per capita e total de domicílios na distribuição de renda |
| `distribuicao_renda` | Quantidade de domicílios em cada faixa de renda (até 1/4 SM, 1/2 SM, 1 SM, 2 SM etc.) |
| `taxa_atividade_pct_cleaned` | Quantidade de ocupados por seção CNAE (agricultura, indústria, comércio, serviços, administração pública etc.) |
| `profissao_cleaned` | Quantidade de pessoas por grande grupo de ocupação (CBO): diretores, técnicos, operários, vendedores etc. |

Tabelas IBGE com metadados no topo usam funções de limpeza documentadas em [`specs/001-censo-streamlit-dashboard/contracts/ibge-table-cleaning.md`](specs/001-censo-streamlit-dashboard/contracts/ibge-table-cleaning.md).

---

## 🔬 Executar o Pipeline

1. Abrir o notebook no Colab — **fonte de verdade**: [link direto](https://colab.research.google.com/drive/1DI1Xzzeo1JjgIgJQQr80zfOTLICUTG3p?usp=sharing) ou File → Open notebook → GitHub → `LuanLindolfo/tcc` (branch `001-censo-streamlit-dashboard`) → `notebooks/censo_castanhal_pipeline.ipynb`
2. Configurar `GITHUB_TOKEN` nos Colab Secrets (🔑 menu lateral)
   - Token precisa de permissão `Contents: Write` no repositório `LuanLindolfo/tcc`
3. Executar todas as células (Ctrl+F9) — os XLSX são baixados automaticamente do GitHub
4. Verificar artefatos gerados em `data/processed/`, `models/` e `data/results/` no GitHub

---

## 🧠 Modelos de ML

| Modelo | Algoritmo | Target | Métricas |
|--------|-----------|--------|---------|
| IVS (Vulnerabilidade) | Random Forest Classifier | baixa/media/alta | Acurácia, F1-macro |
| IAH (Infraestrutura) | XGBoost Regressor | 0.0–1.0 | R², RMSE, MAE |
| Clustering Ocupação | K-Means (k=3) | clusters | Distribuição de setores |

---

## 🔒 Segurança & LGPD

- XLSX do IBGE/Censo 2022 são **dados públicos** — commitados em `censo_castanhal/censo_castanhal/`
- Apenas dados **agregados** vão para `data/processed/` (sem identificação individual)
- Nenhum dado pessoal ou sensível é processado (conformidade LGPD Art. 7º)
- Chaves de API armazenadas exclusivamente em Secrets (Colab/Streamlit)
- Conformidade com a LGPD — Art. 7º e Art. 11º (finalidade pública/acadêmica)

---

## 📦 Dependências Principais

```
streamlit>=1.36  pandas>=2.0  plotly>=5.18  scikit-learn>=1.4
xgboost>=2.0     pyarrow>=14  google-generativeai>=0.5  joblib>=1.3
```

Ver `requirements.txt` para lista completa.

---

## 📋 Quickstart Completo

Ver [`specs/001-censo-streamlit-dashboard/quickstart.md`](specs/001-censo-streamlit-dashboard/quickstart.md)

---

*Dados: IBGE Censo 2010–2022 | IA: Google Gemini (`gemini-2.5-flash` em `utils/gemini_utils.py`) | Deploy: Streamlit Community Cloud*
