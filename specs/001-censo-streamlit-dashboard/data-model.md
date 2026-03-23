# Data Model: Sistema de Inteligência Territorial de Castanhal

**Branch**: `001-censo-streamlit-dashboard` | **Date**: 2026-03-16

---

## Período dos Dados

| Ano | Fonte | Escopo |
|-----|-------|--------|
| **2010** | Censo Demográfico IBGE | Usado **apenas** na taxa de crescimento anual (comparação 2010→2022). XLSX do Censo 2010 não estão no repositório (item diferido). |
| **2022** | Censo Demográfico IBGE | Dados principais — demografia, domicílios, educação, trabalho, renda |

Os dados utilizados no projeto são exclusivamente dos Censos **2010** e **2022** (ou posteriores). Nenhum dado anterior a 2010 é considerado.

### Indicadores que usam Censo 2010

| Indicador | Onde | Descrição |
|-----------|------|-----------|
| **Taxa_Crescimento_Anual** | `df_demografia`, `Censo 2022 - Taxa de crescimento anual da população - Municípios.xlsx` | Taxa média geométrica de crescimento entre **Censo 2010** e **Censo 2022**. O valor já vem calculado pelo IBGE no arquivo do Censo 2022. |

*\* `df_demografia`: 2010+2022 apenas em Taxa_Crescimento_Anual; demais campos são 2022.*

---

## Visão Geral

O sistema opera com dados em duas camadas:
1. **Colab** — transforma XLSX brutos em `.parquet` processados e treina modelos
2. **Streamlit** — lê `.parquet` e `.joblib` do GitHub para visualização e IA

Todos os dados são agregados por **setor censitário** ou **município** (Castanhal). Não há dados individuais.

---

## Entidades de Dados (Arquivos Parquet)

### 1. `demografico.parquet`

Dados demográficos por setor censitário de Castanhal.

| Campo | Tipo | Descrição | Fonte (ano) |
|-------|------|------------|-----------|
| `setor_id` | `str` (PK) | Código do setor censitário | 2022 |
| `bairro` | `str` | Nome do bairro (se disponível) | 2022 |
| `pop_total` | `int` | Total de habitantes | 2022 |
| `densidade_demografica` | `float` | Hab./km² | Calculado |
| `pop_masculina` | `int` | Total de homens | 2022 |
| `pop_feminina` | `int` | Total de mulheres | 2022 |
| `razao_sexo` | `float` | Homens por 100 mulheres | Calculado |
| `pop_0_14` | `int` | Pop. de 0 a 14 anos | 2022 |
| `pop_15_64` | `int` | Pop. de 15 a 64 anos | 2022 |
| `pop_65_mais` | `int` | Pop. de 65+ anos | 2022 |
| `indice_envelhecimento` | `float` | (Pop.65+ / Pop.0-14) × 100 | Calculado |
| `ide` | `float` | Indicador de Dependência Econômica | Calculado (feature composta) |
| `pct_branca` | `float` | % população branca | 2022 |
| `pct_preta` | `float` | % população preta | 2022 |
| `pct_parda` | `float` | % população parda | 2022 |
| `pct_indigena` | `float` | % população indígena | 2022 |
| `pct_amarela` | `float` | % população amarela | 2022 |
| `pct_naturais_castanhal` | `float` | % residentes nascidos em Castanhal | 2022 |
| `pct_migrantes` | `float` | % residentes nascidos fora do Pará | Calculado |

---

### 2. `domicilios.parquet`

Condições habitacionais por setor censitário.

| Campo | Tipo | Descrição | Fonte (ano) |
|-------|------|------------|-----------|
| `setor_id` | `str` (FK → demografico) | Código do setor censitário | 2022 |
| `total_domicilios` | `int` | Total de domicílios ocupados | 2022 |
| `media_moradores` | `float` | Média de moradores/domicílio | Calculado |
| `pct_casas` | `float` | % domicílios do tipo casa | 2022 |
| `pct_apartamentos` | `float` | % domicílios do tipo apartamento | 2022 |
| `pct_comodos` | `float` | % domicílios do tipo cômodo | 2022 |
| `pct_agua_encanada` | `float` | % com acesso à água encanada | 2022 |
| `pct_esgoto` | `float` | % com esgoto sanitário | 2022 |
| `pct_coleta_lixo` | `float` | % com coleta de lixo | 2022 |
| `pct_energia_eletrica` | `float` | % com energia elétrica | 2022 |
| `pct_parede_alvenaria` | `float` | % paredes de alvenaria | 2022 |
| `pct_parede_madeira` | `float` | % paredes de madeira | 2022 |
| `pct_parede_taipa` | `float` | % paredes de taipa | 2022 |
| `pct_proprio` | `float` | % domicílios próprios | 2022 |
| `pct_alugado` | `float` | % domicílios alugados | 2022 |
| `pct_cedido` | `float` | % domicílios cedidos | 2022 |
| `iah` | `float` | Índice de Adequação Habitacional (0–1) | Calculado (feature composta) |

**IAH (feature composta)**:
```
IAH = (pct_agua_encanada + pct_esgoto + pct_coleta_lixo + pct_energia_eletrica) / 4
```
Normalizado para 0–1. Alvo da regressão.

---

### 3. `educacao.parquet`

Indicadores educacionais por setor censitário.

| Campo | Tipo | Descrição | Fonte (ano) |
|-------|------|------------|-----------|
| `setor_id` | `str` (FK) | Código do setor censitário | 2022 |
| `taxa_analfabetismo` | `float` | % analfabetos (15+ anos) | 2022 |
| `pct_fundamental_incompleto` | `float` | % sem fundamental completo | 2022 |
| `pct_fundamental_completo` | `float` | % com fundamental completo | 2022 |
| `pct_medio_incompleto` | `float` | % com médio incompleto | 2022 |
| `pct_medio_completo` | `float` | % com médio completo | 2022 |
| `pct_superior_incompleto` | `float` | % com superior incompleto | 2022 |
| `pct_superior_completo` | `float` | % com superior completo | 2022 |
| `pct_freq_escolar_criancas` | `float` | % crianças (6-14) na escola | 2022 |
| `pct_freq_escolar_jovens` | `float` | % jovens (15-17) na escola | 2022 |
| `escolaridade_media_anos` | `float` | Anos médios de estudo | Calculado |

---

### 4. `trabalho_renda.parquet`

Indicadores de trabalho e renda por setor censitário.

| Campo | Tipo | Descrição | Fonte (ano) |
|-------|------|------------|-----------|
| `setor_id` | `str` (FK) | Código do setor censitário | 2022 |
| `taxa_atividade_pea` | `float` | % PEA sobre pop. em idade ativa | 2022 |
| `pct_setor_primario` | `float` | % PEA em agropecuária/extrativismo | 2022 |
| `pct_setor_secundario` | `float` | % PEA em indústria/construção | 2022 |
| `pct_setor_terciario` | `float` | % PEA em comércio/serviços | 2022 |
| `renda_media_per_capita` | `float` | Renda domiciliar per capita (R$) | 2022 |
| `pct_sem_renda` | `float` | % domicílios sem rendimento | 2022 |
| `pct_ate_1sm` | `float` | % com renda até 1 salário mínimo | 2022 |
| `pct_1_2sm` | `float` | % com renda de 1 a 2 SM | 2022 |
| `pct_acima_2sm` | `float` | % com renda acima de 2 SM | 2022 |
| `indice_gini` | `float` | Índice de Gini — ⏳ **DIFERIDO** (campo reservado no schema; cálculo suspenso até disponibilidade de microdados) | Calculado |
| `cluster_ocupacao` | `int` (0–3) | Cluster K-Means por perfil econômico | ML output |

---

### 5. `features_compostas.parquet`

Dataset consolidado para treinamento ML (join de todas as entidades acima).

| Campo | Tipo | Descrição |
|-------|------|------------|
| `setor_id` | `str` (PK) | Código do setor censitário |
| *(todas as features acima)* | — | Join de demografico + domicilios + educacao + trabalho_renda |
| `iah` | `float` | **Alvo de Regressão** |
| `ivs_label` | `str` | **Alvo de Classificação** (`"baixa"` / `"media"` / `"alta"`) |
| `ivs_cluster` | `int` | Cluster numérico do K-Means (0, 1, 2) |
| `zona_envelhecimento` | `bool` | `True` se `indice_envelhecimento` > 3º quartil |

---

## Entidades de Modelos e Resultados

### 6. `models/random_forest_ivs.joblib`

Modelo serializado para classificação de vulnerabilidade.

| Atributo | Valor |
|----------|-------|
| Tipo | `sklearn.ensemble.RandomForestClassifier` |
| Target | `ivs_label` (baixa/media/alta) |
| Features | Todas colunas exceto `setor_id`, `iah`, `ivs_label`, `ivs_cluster` |
| Serialização | `joblib.dump()` |

### 7. `models/xgboost_iah.joblib`

Modelo serializado para regressão de infraestrutura.

| Atributo | Valor |
|----------|-------|
| Tipo | `xgboost.XGBRegressor` |
| Target | `iah` (0.0 – 1.0) |
| Features | Variáveis demográficas + renda + educação |
| Serialização | `joblib.dump()` |

### 8. `models/kmeans_ocupacao.joblib`

Modelo de clustering para perfil econômico.

| Atributo | Valor |
|----------|-------|
| Tipo | `sklearn.cluster.KMeans` |
| k | 3 ou 4 clusters (definir na análise exploratória) |
| Features | `pct_setor_primario`, `pct_setor_secundario`, `pct_setor_terciario`, `renda_media_per_capita` |

---

## Entidades de Resultados (JSON)

### 9. `data/results/ml_classificacao_metricas.json`

```json
{
  "modelo": "RandomForestClassifier",
  "data_treino": "2026-03-16",
  "acuracia": 0.87,
  "f1_macro": 0.85,
  "classes": ["baixa", "media", "alta"],
  "matriz_confusao": [[10, 2, 0], [1, 15, 3], [0, 2, 12]],
  "feature_importances": {
    "renda_media_per_capita": 0.23,
    "pct_esgoto": 0.18,
    "taxa_analfabetismo": 0.15
  }
}
```

### 10. `data/results/ml_regressao_metricas.json`

```json
{
  "modelo": "XGBRegressor",
  "data_treino": "2026-03-16",
  "r2": 0.82,
  "rmse": 0.08,
  "mae": 0.06
}
```

### 11. `data/results/politicas_recomendacoes.json`

```json
[
  {
    "politica": "Programa de Habitação Popular",
    "area": "domicilios",
    "modelo_relacionado": "xgboost_iah",
    "indicador_chave": "iah",
    "descricao": "Setores com IAH < 0.5 são prioritários para intervenção habitacional.",
    "setores_prioritarios": ["123456789", "987654321"]
  }
]
```

---

## Entidade de Sessão de Chat

### 12. `ChatSession` (estado em memória no Streamlit)

Gerenciado via `st.session_state` — não persistido em arquivo.

| Campo | Tipo | Descrição |
|-------|------|------------|
| `messages` | `list[dict]` | Lista de `{"role": "user"/"assistant", "content": str}` |
| `contexto_ativo` | `str` | Tema atual da conversa (ex: "educacao", "renda") |
| `dados_contexto` | `pd.DataFrame` | Slice do DataFrame relevante para o tema |

---

## Diagrama de Relacionamentos

```
XLSX no Drive (brutos)
        ↓ [Colab: limpeza + feature engineering]
        ↓
demografico.parquet ────┐
domicilios.parquet ─────┼──→ features_compostas.parquet
educacao.parquet ───────┤         ↓ [Colab: treinamento ML]
trabalho_renda.parquet ─┘         ↓
                              models/*.joblib
                              results/*.json
                                   ↓ [GitHub → Streamlit]
                                   ↓
                          Streamlit Dashboard
                          (visualização + IA conversacional)
```

---

---

## Tabelas Consolidadas (tcc_tabelas_merge.ipynb)

O notebook `tcc_tabelas_merge.ipynb` na raiz do repositório gera **DataFrames consolidados por tópico** do documento TCC. **Período dos dados**: 2010–2022 (Censo IBGE). Fonte: Google Drive `/content/drive/MyDrive/censo_castanhal/` ou GitHub fallback.

### Tabelas geradas

| DataFrame | O que é | Ano | Colunas principais |
|------------|---------|-----|--------------------|
| `df_demografia` | Indicadores demográficos do município: população total, densidade, urbanização, taxa de crescimento anual (2010→2022), índice de envelhecimento, razão de sexo | 2010+2022* | Municipio, Codigo_Municipio, Populacao_Total, Densidade_Demografica, Percentual_Urbano, **Taxa_Crescimento_Anual**, Indice_Envelhecimento, Razao_Sexo |
| `df_domicilios` | Características habitacionais: tipos de domicílio (casa/apartamento/cômodo), saneamento (água, esgoto, lixo, energia), material das paredes, condições de ocupação | 2022 | Codigo_Municipio, total_domicilios, pct_casas, pct_apartamentos, saneamento, material_paredes, ocupacao |
| `df_educacao` | Escolaridade por sexo (18+ anos): pessoas por nível de instrução — sem instrução/fundamental incompleto, fundamental completo, médio completo, superior completo | 2022 | Municipio, Sexo, Ano, Total_Pessoas_18_ou_mais, Sem_instr_fund_incompl, Fund_compl_medio_incompl, Medio_compl_super_incompl, Superior_completo |
| `df_educacao_sexo_pct` | Versão percentual de df_educacao (mesmas variáveis com sufixo _Perc) | 2022 | Municipio, Sexo, Ano, Total_Pessoas_18_ou_mais_Perc, ... |
| `df_trabalho_renda` | Condição de atividade econômica (PEA): economicamente ativas (ocupadas/desocupadas) e não economicamente ativas | 2022 | Condicao_Atividade, Total, Codigo_Municipio |
| `taxa_atividade_pct_cleaned` | Distribuição de ocupados por seção CNAE (agricultura, indústria, comércio, construção, administração pública, educação etc.) | 2022 | Seção de Atividade, Valor |
| `profissao_cleaned` | Distribuição por grande grupo de ocupação CBO (diretores, técnicos, vendedores, operários, ocupações elementares etc.) | 2022 | Grupo de Ocupação, Valor |
| `df_renda` | Rendimento domiciliar mensal per capita (R$) e total de domicílios com dados de distribuição de renda | 2022 | Codigo_Municipio, Municipio, Rendimento_Per_Capita, Total_Domicilios_DistRenda |
| `distribuicao_renda` | Quantidade de domicílios em cada faixa de renda (até 1/4 SM, 1/2 SM, 1 SM, 2 SM, 3 SM ... até sem rendimento) | 2022 | Classes_de_Rendimento, Total |

**Funções de limpeza**: Ver [contracts/ibge-table-cleaning.md](./contracts/ibge-table-cleaning.md).

---

## Regras de Validação

| Regra | Campo | Constraint |
|-------|-------|------------|
| Setor único | `setor_id` | PK único em cada parquet |
| Percentuais válidos | Todos `pct_*` | `0.0 ≤ valor ≤ 100.0` |
| IAH normalizado | `iah` | `0.0 ≤ valor ≤ 1.0` |
| IVS válido | `ivs_label` | Apenas `"baixa"`, `"media"`, `"alta"` |
| Renda não-negativa | `renda_media_per_capita` | `valor ≥ 0` |
| Gini válido | `indice_gini` | `0.0 ≤ valor ≤ 1.0` |
| NaN handling | Todos campos ML | Imputar mediana antes do treinamento |
