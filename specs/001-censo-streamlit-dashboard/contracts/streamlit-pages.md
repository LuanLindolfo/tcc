# Contrato de Interface: Páginas Streamlit

**Projeto**: Sistema de Inteligência Territorial de Castanhal  
**Data**: 2026-03-16

---

## Estrutura de Navegação

O app usa `st.navigation` (Streamlit 1.32+) com páginas definidas como arquivos em `pages/`. A sidebar exibe o menu de navegação sempre visível.

```
app.py                    ← Entry point principal
pages/
├── 1_demografia.py       ← Aba 1: Dinâmica Populacional
├── 2_domicilios.py       ← Aba 2: Diagnóstico Habitacional
├── 3_educacao_renda.py   ← Aba 3: Educação & Trabalho/Renda
├── 4_machine_learning.py ← Aba 4: Resultados de ML
├── 5_politicas.py        ← Aba 5: Políticas Públicas
└── 6_assistente_ia.py    ← Aba 6: Assistente IA (Gemini)
utils/
├── data_loader.py        ← Funções de carga de dados do GitHub
├── ml_utils.py           ← Carregamento de modelos e predições
└── gemini_utils.py       ← Integração com Google Gemini
```

---

## Contrato por Página

### Página 1: Dinâmica Populacional (`1_demografia.py`)

**Entradas**:
- Dados: `demografico.parquet`
- Filtros: nenhum obrigatório (exibe município todo por padrão)

**Componentes obrigatórios**:
| Componente | Tipo Streamlit | Dado |
|-----------|---------------|------|
| Pirâmide etária | `plotly` bar chart horizontal | `pop_0_14`, `pop_15_64`, `pop_65_mais` por sexo |
| Distribuição étnico-racial | `plotly` pie chart | `pct_*` raça |
| Mapa de densidade | `plotly` choropleth ou `st.map` | `densidade_demografica` por setor |
| KPIs | `st.metric` | `pop_total`, `indice_envelhecimento`, `razao_sexo` |
| Crescimento histórico | `plotly` line chart | Comparação Censo 2010 vs 2022 (se disponível) |

**Estado de erro**: Se `demografico.parquet` indisponível → exibir `st.error("Dados não disponíveis. Execute o pipeline no Colab.")`

---

### Página 2: Diagnóstico Habitacional (`2_domicilios.py`)

**Entradas**:
- Dados: `domicilios.parquet`

**Componentes obrigatórios**:
| Componente | Tipo Streamlit | Dado |
|-----------|---------------|------|
| Mapa de calor IAH | `plotly` choropleth | `iah` por setor |
| Saneamento básico | `plotly` bar chart | `pct_agua_encanada`, `pct_esgoto`, `pct_coleta_lixo`, `pct_energia_eletrica` |
| Tipos de domicílio | `plotly` donut chart | `pct_casas`, `pct_apartamentos`, `pct_comodos` |
| Posse e ocupação | `plotly` bar chart | `pct_proprio`, `pct_alugado`, `pct_cedido` |
| KPIs | `st.metric` | `media_moradores`, % com todos serviços básicos |

---

### Página 3: Educação & Trabalho/Renda (`3_educacao_renda.py`)

**Entradas**:
- Dados: `educacao.parquet` + `trabalho_renda.parquet`

**Componentes obrigatórios**:
| Componente | Tipo Streamlit | Dado |
|-----------|---------------|------|
| Pirâmide educacional | `plotly` funnel chart | Todos `pct_*` níveis de instrução |
| Dispersão renda × escolaridade | `plotly` scatter | `escolaridade_media_anos` vs `renda_media_per_capita` |
| Distribuição de renda | `plotly` histogram | `renda_media_per_capita` por setor |
| Gini por setor | `plotly` bar chart | `indice_gini` (se disponível) |
| PEA por setor econômico | `plotly` stacked bar | `pct_setor_primario/secundario/terciario` |
| KPIs | `st.metric` | `taxa_analfabetismo`, renda média municipal, taxa de frequência escolar |

---

### Página 4: Resultados de ML (`4_machine_learning.py`)

**Entradas**:
- Dados: `features_compostas.parquet`
- Modelos: `random_forest_ivs.joblib`, `xgboost_iah.joblib`, `kmeans_ocupacao.joblib`
- Métricas: `ml_classificacao_metricas.json`, `ml_regressao_metricas.json`

**Componentes obrigatórios**:
| Componente | Tipo Streamlit | Dado |
|-----------|---------------|------|
| Métricas de classificação | `st.metric` + tabela | Acurácia, F1-macro |
| Matriz de confusão | `plotly` heatmap | `matriz_confusao` do JSON |
| Feature importance | `plotly` bar chart horizontal | `feature_importances` do JSON |
| Mapa de IVS | `plotly` choropleth | `ivs_label` por setor |
| Métricas de regressão | `st.metric` | R², RMSE, MAE |
| Gráfico real vs predito | `plotly` scatter | IAH real vs IAH predito |
| Clusters de ocupação | `plotly` scatter 2D | `cluster_ocupacao` colorido |

**Estado sem artefatos**: `st.warning("Modelos ainda não treinados. Execute o notebook Colab para gerar os artefatos de ML.")`

---

### Página 5: Políticas Públicas (`5_politicas.py`)

**Entradas**:
- Dados: `politicas_recomendacoes.json` + `features_compostas.parquet`

**Componentes obrigatórios**:
| Componente | Tipo Streamlit | Dado |
|-----------|---------------|------|
| Seletor de área temática | `st.selectbox` | ["Habitação", "Educação", "Saúde", "Trabalho & Renda", "Infraestrutura"] |
| Card de política | `st.expander` | Descrição + indicador-chave + modelo ML relacionado |
| Mapa de setores prioritários | `plotly` choropleth | `setores_prioritarios` do JSON |
| Tabela de setores críticos | `st.dataframe` | Top 10 setores por indicador selecionado |
| Recomendação de ação | `st.info` | Texto da recomendação baseada em ML |

**Políticas mínimas (3 obrigatórias)**:
1. **Habitação Popular** — baseada em `iah` < 0.5 (modelo regressão)
2. **Combate ao Analfabetismo** — baseada em `taxa_analfabetismo` > quartil 75 (dados diretos)
3. **Geração de Emprego** — baseada em `cluster_ocupacao` + `taxa_atividade_pea` (modelo clustering)

---

### Página 6: Assistente IA (`6_assistente_ia.py`)

**Entradas**:
- Todos os parquets (para contexto dinâmico)
- `st.secrets["GEMINI_API_KEY"]`

**Componentes obrigatórios**:
| Componente | Tipo Streamlit | Descrição |
|-----------|---------------|-----------|
| Histórico de mensagens | Loop sobre `st.session_state.messages` | Exibe mensagens user/assistant |
| Input de pergunta | `st.chat_input` | Campo de digitação na base da tela |
| Seletor de tema | `st.selectbox` (sidebar) | Pré-carrega contexto de dados relevante |
| Indicador de loading | `st.spinner` | Exibido enquanto Gemini processa |

**Comportamento**:
1. Usuário seleciona tema (ex: "Educação")
2. Sistema injeta resumo estatístico do `educacao.parquet` no prompt
3. Usuário digita pergunta
4. Gemini responde com contexto dos dados
5. Troca adicionada ao histórico visível

**Estado sem API key**: `st.error("Chave da API Gemini não configurada. Adicione GEMINI_API_KEY nos Secrets do Streamlit.")`

---

## Contrato de Dados Compartilhados (`utils/data_loader.py`)

```python
# Assinaturas de funções públicas obrigatórias

def get_base_url() -> str:
    """Retorna URL base do repositório GitHub."""

def load_demografico() -> pd.DataFrame:
    """Carrega demografico.parquet com cache de 1 hora."""

def load_domicilios() -> pd.DataFrame:
    """Carrega domicilios.parquet com cache de 1 hora."""

def load_educacao() -> pd.DataFrame:
    """Carrega educacao.parquet com cache de 1 hora."""

def load_trabalho_renda() -> pd.DataFrame:
    """Carrega trabalho_renda.parquet com cache de 1 hora."""

def load_features_compostas() -> pd.DataFrame:
    """Carrega features_compostas.parquet com cache de 1 hora."""

def load_politicas() -> list[dict]:
    """Carrega politicas_recomendacoes.json."""

def load_ml_metricas(tipo: Literal["classificacao", "regressao"]) -> dict:
    """Carrega métricas do modelo especificado."""
```

**Decorador obrigatório**: `@st.cache_data(ttl=3600)` em todas as funções de carga.

**Tratamento de erro padrão**:
```python
try:
    return pd.read_parquet(url)
except Exception:
    st.error(f"Erro ao carregar dados de {url}")
    return pd.DataFrame()  # DataFrame vazio como fallback
```
