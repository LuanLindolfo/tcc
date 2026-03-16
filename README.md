# 🏙️ Sistema de Inteligência Territorial de Castanhal

Dashboard interativo para análise dos dados do **Censo IBGE 2022** do município de
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

| Página | Conteúdo |
|--------|----------|
| 📊 Dinâmica Populacional | Pirâmide etária, distribuição étnico-racial, densidade |
| 🏠 Diagnóstico Habitacional | IAH, saneamento básico, tipos de domicílio |
| 📚 Educação & Renda | Escolaridade, distribuição de renda, PEA |
| 🤖 Machine Learning | Classificação IVS, Regressão IAH, Clustering |
| 🏛️ Políticas Públicas | 5 políticas com setores prioritários por ML |
| 💬 Assistente IA | Chat contextualizado com Google Gemini |

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
- `GITHUB_RAW_BASE` → `https://raw.githubusercontent.com/LuanLindolfo/tcc/main`

**Google Colab** (pipeline):
- `GITHUB_TOKEN` → Personal Access Token com permissão `Contents: Write`

---

## 🔬 Executar o Pipeline

1. Abrir `notebooks/censo_castanhal_pipeline.ipynb` no Google Colab
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
streamlit>=1.32  pandas>=2.0  plotly>=5.18  scikit-learn>=1.4
xgboost>=2.0     pyarrow>=14  google-generativeai>=0.5  joblib>=1.3
```

Ver `requirements.txt` para lista completa.

---

## 📋 Quickstart Completo

Ver [`specs/001-censo-streamlit-dashboard/quickstart.md`](specs/001-censo-streamlit-dashboard/quickstart.md)

---

*Dados: IBGE Censo 2022 | IA: Google Gemini 1.5 Flash | Deploy: Streamlit Community Cloud*
