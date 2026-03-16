# Tasks: Sistema de Inteligência Territorial de Castanhal

**Feature**: `001-censo-streamlit-dashboard`  
**Gerado em**: 2026-03-16  
**Colab (fonte de verdade)**: https://colab.research.google.com/drive/1DI1Xzzeo1JjgIgJQQr80zfOTLICUTG3p?usp=sharing  
**Repositório**: https://github.com/LuanLindolfo/tcc  
**Total de tarefas**: 22 | **Total de fases**: 7

> **Contexto atual**: Estrutura Streamlit criada localmente. XLSX do IBGE já estão no GitHub em
> `censo_castanhal/censo_castanhal/`. Notebook do Colab é a **fonte de verdade** do pipeline.
> Pasta local `/home/lindolfo/tcc` ainda NÃO foi commitada/pushed para o GitHub.

---

## Arquitetura do Pipeline (fonte de verdade = Colab)

```
GitHub (censo_castanhal/censo_castanhal/*.xlsx — dados IBGE públicos)
      ↓  lidos via URL raw pelo notebook no Colab
Google Colab (fonte de verdade: limpa + treina ML + gera artefatos)
      ↓  push via GITHUB_TOKEN (Colab Secrets)
GitHub (data/processed/*.parquet + models/*.joblib + data/results/*.json)
      ↓  GitHub raw URLs + @st.cache_data
Streamlit Community Cloud (visualização pública — banca avaliadora)
```

---

## Dependências entre Fases

```
Fase 1 (Git/Push)
    ↓
Fase 2 (Notebook Colab atualizado)
    ↓
Fase 3 (Pipeline end-to-end)
    ↓
Fase 4 (US1 — Streamlit dados)  ←→  Fase 5 (US2 — ML)
    ↓                                        ↓
Fase 6 (US3 — Gemini IA)         Fase 6 (US4 — Políticas)
    ↓
Fase 7 (Deploy)
```

---

## Fase 1: Setup — Publicar código local no GitHub

> Objetivo: tornar o repositório local `/home/lindolfo/tcc` o repositório GitHub, para que
> o Streamlit Community Cloud possa fazer deploy a partir dele.

- [x] T001 Inicializar git no workspace local: `git init && git remote add origin https://github.com/LuanLindolfo/tcc.git` em `/home/lindolfo/tcc`
- [x] T002 Fazer primeiro commit e push de toda a estrutura Streamlit para a branch main do GitHub: `git add . && git commit -m "feat: add streamlit app structure"` em `/home/lindolfo/tcc`
- [ ] T003 Verificar no GitHub que os seguintes arquivos existem após o push: `app.py`, `requirements.txt`, `pages/`, `utils/`, `notebooks/censo_castanhal_pipeline.ipynb`

---

## Fase 2: US5 — Notebook Colab como Fonte de Verdade

> Objetivo: garantir que o notebook no Colab (https://colab.research.google.com/drive/1DI1Xzzeo1JjgIgJQQr80zfOTLICUTG3p)
> usa o pipeline atualizado que lê XLSX do GitHub (sem Google Drive).

- [ ] T004 [US5] No Colab, abrir o notebook da URL e substituir seu conteúdo pelo arquivo `notebooks/censo_castanhal_pipeline.ipynb` do GitHub (File → Open Notebook → GitHub → LuanLindolfo/tcc → notebooks/censo_castanhal_pipeline.ipynb), OU copiar o conteúdo célula a célula
- [ ] T005 [US5] Configurar o secret `GITHUB_TOKEN` no Colab Secrets (🔑 menu lateral): Personal Access Token com permissão `Contents: Write` no repositório `LuanLindolfo/tcc` — gerado em https://github.com/settings/tokens
- [ ] T006 [US5] Executar apenas a **Célula 2** do notebook e inspecionar o output: anotar os nomes de colunas reais retornados para cada um dos 19 XLSX (o print mostrará as colunas depois do `snake()`)
- [ ] T007 [US5] Atualizar a **Célula 4** do notebook (consolidação) com os nomes de colunas reais detectados na T006 — mapear cada coluna para o campo esperado (ex: coluna de faixas etárias, coluna de sexo, coluna de valor de renda, etc.) em `notebooks/censo_castanhal_pipeline.ipynb`

---

## Fase 3: US5 — Pipeline End-to-End (Colab → GitHub)

> Critério independente: Executar notebook gera artefatos no GitHub que o Streamlit lê.

- [ ] T008 [US5] Executar o notebook completo (Ctrl+F9) no Colab e verificar que todos os artefatos aparecem no GitHub: `data/processed/demografico.parquet`, `data/processed/domicilios.parquet`, `data/processed/educacao.parquet`, `data/processed/trabalho_renda.parquet`, `data/processed/features_compostas.parquet`
- [ ] T009 [US5] Verificar que os modelos e métricas foram gerados no GitHub: `models/random_forest_ivs.joblib`, `models/xgboost_iah.joblib`, `models/kmeans_ocupacao.joblib`, `data/results/ml_classificacao_metricas.json`, `data/results/ml_regressao_metricas.json`, `data/results/politicas_recomendacoes.json`
- [ ] T010 [US5] Commitar e fazer push do notebook atualizado com os mapeamentos de colunas reais: `git add notebooks/ && git commit -m "feat: update pipeline with real column mappings"` em `/home/lindolfo/tcc`

---

## Fase 4: US1 — Streamlit: Adaptar Páginas às Colunas Reais

> Critério independente: Todas as 3 abas de dados carregam gráficos corretos sem erro.

- [x] T011 [P] [US1] Atualizar `utils/data_loader.py` com as funções `carregar_demografico()`, `carregar_domicilios()`, `carregar_educacao()`, `carregar_trabalho_renda()` e `carregar_features()` usando os nomes de colunas reais confirmados na T006 — garantir que cada função retorna DataFrame não-vazio com as colunas esperadas pelos gráficos
- [x] T012 [P] [US1] Atualizar `pages/1_demografia.py` para usar as colunas reais da pirâmide etária (ex: coluna de faixa etária, colunas de homens/mulheres) e do arquivo de razão de sexo/índice de envelhecimento, garantindo que os gráficos renderizam sem `KeyError`
- [x] T013 [P] [US1] Atualizar `pages/2_domicilios.py` para usar colunas reais de saneamento, tipos de domicílio e condições de ocupação dos XLSX — garantir que KPIs e histograma IAH exibem valores corretos
- [x] T014 [P] [US1] Atualizar `pages/3_educacao_renda.py` para usar colunas reais de nível de instrução, taxa de analfabetismo, frequência escolar e rendimento per capita — placeholder do Gini deve permanecer com `st.info()`

---

## Fase 5: US2 — Resultados de ML no Streamlit

> Critério independente: Aba ML exibe métricas e gráficos interpretáveis carregados do GitHub.

- [ ] T015 [P] [US2] Validar `pages/4_machine_learning.py` após execução do pipeline (Fase 3): verificar que as 3 abas internas (Classificação, Regressão, Clustering) carregam métricas dos JSONs e gráficos sem erro — corrigir quaisquer referências a colunas inexistentes em `pages/4_machine_learning.py`
- [x] T016 [US2] Adicionar filtro interativo por cluster (`st.selectbox`) na aba de Clustering em `pages/4_machine_learning.py` para satisfazer US2 Acceptance Scenario 2 (filtro que atualiza gráfico) — executar após T015

---

## Fase 6: US3 + US4 — Gemini IA e Políticas Públicas

> Critério US3: Resposta coerente a 3 perguntas sobre os dados via Gemini.
> Critério US4: Aba exibe 5 políticas com setores prioritários identificados pelos modelos.

- [ ] T017 [P] [US3] Obter `GEMINI_API_KEY` em https://aistudio.google.com/app/apikey e configurar localmente em `.streamlit/secrets.toml` (nunca commitar) — testar resposta no `pages/6_assistente_ia.py` rodando `streamlit run app.py` localmente
- [ ] T018 [P] [US4] Validar `pages/5_politicas.py` após pipeline executado (Fase 3): verificar que o JSON `politicas_recomendacoes.json` é lido corretamente e os 5 cards de política são exibidos sem erro

---

## Fase 7: Deploy — Streamlit Community Cloud

> Critério final: URL pública funcional, Streamlit lê dados do GitHub, banca avaliadora acessa.

- [ ] T019 Acessar https://share.streamlit.io → "New app" → selecionar repositório `LuanLindolfo/tcc` → Main file path: `app.py` → branch: `main` → clicar "Deploy"
- [ ] T020 Configurar os 2 secrets no painel do Streamlit Cloud (Settings → Secrets) conforme template `.streamlit/secrets.toml.example`: `GEMINI_API_KEY` e `GITHUB_RAW_BASE = "https://raw.githubusercontent.com/LuanLindolfo/tcc/main"`
- [ ] T021 Verificar que o app carrega sem erros no Streamlit Cloud — checar logs se alguma página travar e corrigir imports ou colunas faltantes
- [ ] T022 Commitar o notebook final e a URL do app Streamlit no `README.md` do repositório: `git add README.md notebooks/ && git commit -m "docs: add streamlit app URL and finalize pipeline"` em `/home/lindolfo/tcc`

---

## Execução em Paralelo

Tasks com `[P]` na mesma fase podem ser executadas simultaneamente:

| Fase | Paralelas | Sequenciais |
|------|-----------|-------------|
| Fase 4 | T011, T012, T013, T014 (páginas independentes) | após T008-T009 (artefatos no GitHub) |
| Fase 5 | T015 (validação) | T016 sequencial após T015 (mesmo arquivo) |
| Fase 6 | T017 (Gemini), T018 (Políticas) | independentes entre si |

---

## Checklist Rápido por User Story

| Story | Critério | Tasks |
|-------|----------|-------|
| US5 — Pipeline | Colab executa, artefatos aparecem no GitHub | T004–T010 |
| US1 — Dados | 3 abas com gráficos corretos | T011–T014 |
| US2 — ML | Métricas + gráficos + filtro | T015–T016 |
| US3 — IA | Resposta coerente ao chat | T017 |
| US4 — Políticas | 5 políticas com setores | T018 |
| Deploy | URL pública funcionando | T019–T022 |

---

## Localização do GITHUB_TOKEN

> **Onde colocar o token** (resumo para referência rápida):
>
> | Onde | Variável | Como |
> |------|----------|------|
> | **Google Colab** | `GITHUB_TOKEN` | 🔑 Secrets (menu lateral) → Add secret |
> | **Streamlit Cloud** | `GEMINI_API_KEY` | App Settings → Secrets → colar TOML |
> | **Local** | `GEMINI_API_KEY` | `.streamlit/secrets.toml` (não commitar) |
>
> ⚠️ **Nunca** colocar tokens diretamente no código ou commitar `.env` / `secrets.toml`

---

## MVP Sugerido

Execute nesta ordem para ter o dashboard funcional o mais rápido possível:

1. **T001–T003** → código no GitHub
2. **T004–T005** → notebook do Colab configurado com token
3. **T006–T009** → pipeline executa e artefatos aparecem no GitHub
4. **T019–T021** → deploy imediato (já funciona com dados básicos)
5. **T011–T018** → refinamento das páginas com colunas reais
