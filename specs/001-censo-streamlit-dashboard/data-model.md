# Data Model: Sistema de Inteligência Territorial de Castanhal

**Branch**: `001-censo-streamlit-dashboard` | **Date**: 2026-03-16

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

| Campo | Tipo | Descrição | Fonte XLSX |
|-------|------|-----------|-----------|
| `setor_id` | `str` (PK) | Código do setor censitário | Censo 2022 |
| `bairro` | `str` | Nome do bairro (se disponível) | Censo 2022 |
| `pop_total` | `int` | Total de habitantes | Censo 2022 |
| `densidade_demografica` | `float` | Hab./km² | Calculado |
| `pop_masculina` | `int` | Total de homens | Censo 2022 |
| `pop_feminina` | `int` | Total de mulheres | Censo 2022 |
| `razao_sexo` | `float` | Homens por 100 mulheres | Calculado |
| `pop_0_14` | `int` | Pop. de 0 a 14 anos | Censo 2022 |
| `pop_15_64` | `int` | Pop. de 15 a 64 anos | Censo 2022 |
| `pop_65_mais` | `int` | Pop. de 65+ anos | Censo 2022 |
| `indice_envelhecimento` | `float` | (Pop.65+ / Pop.0-14) × 100 | Calculado |
| `ide` | `float` | Indicador de Dependência Econômica | Calculado (feature composta) |
| `pct_branca` | `float` | % população branca | Censo 2022 |
| `pct_preta` | `float` | % população preta | Censo 2022 |
| `pct_parda` | `float` | % população parda | Censo 2022 |
| `pct_indigena` | `float` | % população indígena | Censo 2022 |
| `pct_amarela` | `float` | % população amarela | Censo 2022 |
| `pct_naturais_castanhal` | `float` | % residentes nascidos em Castanhal | Censo 2022 |
| `pct_migrantes` | `float` | % residentes nascidos fora do Pará | Calculado |

---

### 2. `domicilios.parquet`

Condições habitacionais por setor censitário.

| Campo | Tipo | Descrição | Fonte XLSX |
|-------|------|-----------|-----------|
| `setor_id` | `str` (FK → demografico) | Código do setor censitário | Censo 2022 |
| `total_domicilios` | `int` | Total de domicílios ocupados | Censo 2022 |
| `media_moradores` | `float` | Média de moradores/domicílio | Calculado |
| `pct_casas` | `float` | % domicílios do tipo casa | Censo 2022 |
| `pct_apartamentos` | `float` | % domicílios do tipo apartamento | Censo 2022 |
| `pct_comodos` | `float` | % domicílios do tipo cômodo | Censo 2022 |
| `pct_agua_encanada` | `float` | % com acesso à água encanada | Censo 2022 |
| `pct_esgoto` | `float` | % com esgoto sanitário | Censo 2022 |
| `pct_coleta_lixo` | `float` | % com coleta de lixo | Censo 2022 |
| `pct_energia_eletrica` | `float` | % com energia elétrica | Censo 2022 |
| `pct_parede_alvenaria` | `float` | % paredes de alvenaria | Censo 2022 |
| `pct_parede_madeira` | `float` | % paredes de madeira | Censo 2022 |
| `pct_parede_taipa` | `float` | % paredes de taipa | Censo 2022 |
| `pct_proprio` | `float` | % domicílios próprios | Censo 2022 |
| `pct_alugado` | `float` | % domicílios alugados | Censo 2022 |
| `pct_cedido` | `float` | % domicílios cedidos | Censo 2022 |
| `iah` | `float` | Índice de Adequação Habitacional (0–1) | Calculado (feature composta) |

**IAH (feature composta)**:
```
IAH = (pct_agua_encanada + pct_esgoto + pct_coleta_lixo + pct_energia_eletrica) / 4
```
Normalizado para 0–1. Alvo da regressão.

---

### 3. `educacao.parquet`

Indicadores educacionais por setor censitário.

| Campo | Tipo | Descrição | Fonte XLSX |
|-------|------|-----------|-----------|
| `setor_id` | `str` (FK) | Código do setor censitário | Censo 2022 |
| `taxa_analfabetismo` | `float` | % analfabetos (15+ anos) | Censo 2022 |
| `pct_fundamental_incompleto` | `float` | % sem fundamental completo | Censo 2022 |
| `pct_fundamental_completo` | `float` | % com fundamental completo | Censo 2022 |
| `pct_medio_incompleto` | `float` | % com médio incompleto | Censo 2022 |
| `pct_medio_completo` | `float` | % com médio completo | Censo 2022 |
| `pct_superior_incompleto` | `float` | % com superior incompleto | Censo 2022 |
| `pct_superior_completo` | `float` | % com superior completo | Censo 2022 |
| `pct_freq_escolar_criancas` | `float` | % crianças (6-14) na escola | Censo 2022 |
| `pct_freq_escolar_jovens` | `float` | % jovens (15-17) na escola | Censo 2022 |
| `escolaridade_media_anos` | `float` | Anos médios de estudo | Calculado |

---

### 4. `trabalho_renda.parquet`

Indicadores de trabalho e renda por setor censitário.

| Campo | Tipo | Descrição | Fonte XLSX |
|-------|------|-----------|-----------|
| `setor_id` | `str` (FK) | Código do setor censitário | Censo 2022 |
| `taxa_atividade_pea` | `float` | % PEA sobre pop. em idade ativa | Censo 2022 |
| `pct_setor_primario` | `float` | % PEA em agropecuária/extrativismo | Censo 2022 |
| `pct_setor_secundario` | `float` | % PEA em indústria/construção | Censo 2022 |
| `pct_setor_terciario` | `float` | % PEA em comércio/serviços | Censo 2022 |
| `renda_media_per_capita` | `float` | Renda domiciliar per capita (R$) | Censo 2022 |
| `pct_sem_renda` | `float` | % domicílios sem rendimento | Censo 2022 |
| `pct_ate_1sm` | `float` | % com renda até 1 salário mínimo | Censo 2022 |
| `pct_1_2sm` | `float` | % com renda de 1 a 2 SM | Censo 2022 |
| `pct_acima_2sm` | `float` | % com renda acima de 2 SM | Censo 2022 |
| `indice_gini` | `float` | Índice de Gini — ⏳ **DIFERIDO** (campo reservado no schema; cálculo suspenso até disponibilidade de microdados) | Calculado |
| `cluster_ocupacao` | `int` (0–3) | Cluster K-Means por perfil econômico | ML output |

---

### 5. `features_compostas.parquet`

Dataset consolidado para treinamento ML (join de todas as entidades acima).

| Campo | Tipo | Descrição |
|-------|------|-----------|
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
|-------|------|-----------|
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

## Regras de Validação

| Regra | Campo | Constraint |
|-------|-------|-----------|
| Setor único | `setor_id` | PK único em cada parquet |
| Percentuais válidos | Todos `pct_*` | `0.0 ≤ valor ≤ 100.0` |
| IAH normalizado | `iah` | `0.0 ≤ valor ≤ 1.0` |
| IVS válido | `ivs_label` | Apenas `"baixa"`, `"media"`, `"alta"` |
| Renda não-negativa | `renda_media_per_capita` | `valor ≥ 0` |
| Gini válido | `indice_gini` | `0.0 ≤ valor ≤ 1.0` |
| NaN handling | Todos campos ML | Imputar mediana antes do treinamento |
