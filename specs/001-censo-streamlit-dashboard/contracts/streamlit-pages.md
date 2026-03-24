# Contrato de Interface: Páginas Streamlit

**Projeto**: Sistema de Inteligência Territorial de Castanhal  
**Data**: 2026-03-16  
**Atualizado**: 2026-03-24 — alinhado ao app monolítico (`app.py` + `st.navigation`)

---

## Estrutura de Navegação

O app usa **`st.navigation`** (Streamlit 1.36+) no arquivo **`app.py`**. As seções são **funções Python** (`render_*`), não arquivos em `pages/`. Quando `st.navigation` é executado, o Streamlit **ignora** a pasta `pages/` (se existir).

```
app.py                    ← Entry point único: st.navigation + pg.run()
utils/
├── data_loader.py        ← Funções de carga de dados do GitHub
├── ml_utils.py           ← Carregamento de modelos e predições
└── gemini_utils.py       ← Integração com Google Gemini
```

**Seções expostas no menu** (títulos como no `st.Page`):

| Ordem | Título no menu | Função em `app.py` |
|-------|----------------|-------------------|
| 1 | Início | `render_inicio` |
| 2 | Demografia | `render_demografia` |
| 3 | Domicílios | `render_domicilios` |
| 4 | Educação & Renda | `render_educacao_renda` |
| 5 | Políticas | `render_politicas` |
| 6 | Assistente IA | `render_assistente_ia` |

> **Machine Learning**: não há seção dedicada exclusivamente a resultados de ML na UI atual. Artefatos (`features_compostas.parquet`, métricas JSON, modelos `.joblib`) são gerados pelo Colab e **consumidos** pela seção **Políticas** e pelo contexto injetado no **Assistente IA** (`load_features_compostas`, `gerar_contexto_tematico`).

---

## Contrato por Seção

### Início (`render_inicio`)

- Landing com descrição do dashboard e métricas resumo.
- Não carrega parquets obrigatórios.

### Demografia (`render_demografia`)

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

### Domicílios (`render_domicilios`)

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

### Educação & Trabalho/Renda (`render_educacao_renda`)

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

### ~~Resultados de ML~~ (não exposto como seção própria)

O contrato anterior previa `4_machine_learning.py` com métricas, matriz de confusão, feature importance, IVS, regressão IAH e clustering. **Na UI atual essa seção não existe**; o pipeline Colab continua gerando artefatos para o repositório.

---

### Políticas Públicas (`render_politicas`)

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

### Assistente IA (`render_assistente_ia`)

**Entradas**:
- Todos os parquets (para contexto dinâmico)
- `st.secrets["GEMINI_API_KEY"]`

**Componentes obrigatórios**:
| Componente | Tipo Streamlit | Descrição |
|-----------|---------------|-----------|
| Histórico de mensagens | Loop sobre `st.session_state.messages` | Exibe mensagens user/assistant |
| Input de pergunta | `st.chat_input` | Campo de digitação na base da tela |
| Seletor de tema | `st.selectbox` (sidebar, ao visualizar esta seção) | Pré-carrega contexto de dados relevante |
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
