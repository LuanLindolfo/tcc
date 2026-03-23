# Feature Specification: Dashboard Censo Castanhal — ML + IA + Políticas Públicas

**Feature Branch**: `001-censo-streamlit-dashboard`  
**Created**: 2026-03-16  
**Status**: Draft  
**Colab Source**: https://colab.research.google.com/drive/1DI1Xzzeo1JjgIgJQQr80zfOTLICUTG3p?usp=sharing

## Contexto

> **Constituição aplicada**: v1.1.0 — Princípios I–V. O notebook Colab (link acima) é a **fonte de verdade** do pipeline; o arquivo no repo é espelho.
> **Índice de Gini**: ⏳ diferido — campo reservado no schema, cálculo e exibição pendentes.

Projeto de TCC que utiliza dados do **Censo IBGE 2010–2022** do município de **Castanhal – PA**. O período dos dados é restrito a 2010 em diante; dados anteriores não são considerados. O pipeline integra Google Colab (processamento) → GitHub (armazenamento dos xlsx e código) → Streamlit (visualização e exploração interativa). O objetivo final é subsidiar análises de **políticas públicas municipais** com apoio de modelos de Machine Learning (classificação e regressão) e um assistente de IA conversacional.

## Domínio de Dados (Censo Castanhal)

### Notebooks de Dados

| Notebook | Propósito | Localização |
|----------|-----------|-------------|
| `notebooks/censo_castanhal_pipeline.ipynb` | Pipeline Colab: limpeza + ML + push GitHub | Fonte de verdade (Colab); espelho no repo |
| `tcc_tabelas_merge.ipynb` | Tabelas consolidadas por tópico (doc TCC) | Raiz do repo |
| `tcc_censo_2022.ipynb` | Análises exploratórias e processamento auxiliar | Raiz do repo |

O `tcc_tabelas_merge.ipynb` gera: `df_demografia`, `df_domicilios`, `df_educacao`, `df_trabalho_renda`, `df_renda`, `distribuicao_renda`. Tabelas IBGE com metadados no topo usam funções de limpeza documentadas em [contracts/ibge-table-cleaning.md](./contracts/ibge-table-cleaning.md).

### Variáveis disponíveis nos arquivos XLSX importados no Colab:

**Demografia:**
- Número total de habitantes, densidade demográfica
- Distribuição por bairros/setores censitários
- Taxa de crescimento populacional (comparação com censos anteriores)
- Distribuição por faixas etárias, índice de envelhecimento, razão de sexo, pirâmide etária
- Distribuição por cor/raça (étnico-racial), comparação histórica
- Migração: naturalidade dos residentes, fluxos migratórios

**Domicílios:**
- Tipos de domicílio (casas, apartamentos, cômodos)
- Condições de moradia: moradores/domicílio, acesso a água, esgoto, coleta de lixo, energia elétrica
- Materiais predominantes nas paredes
- Posse/condição de ocupação (próprio, alugado, cedido)

**Educação:**
- Distribuição por nível de instrução (fundamental, médio, superior — completo/incompleto)
- Taxa de analfabetismo
- Escolaridade por faixa etária e gênero
- Frequência escolar de crianças e jovens

**Trabalho e Renda:**
- PEA: taxa de atividade, distribuição por setor (primário, secundário, terciário), tipos de ocupação
- Renda média domiciliar per capita, distribuição de renda
- Índice de Gini — ⏳ **DIFERIDO** (campo reservado; implementação futura confirmada pelo autor)
- Renda por escolaridade e ocupação

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Exploração Visual dos Dados do Censo (Priority: P1)

Como pesquisador/gestor público, quero explorar graficamente os dados demográficos, de domicílios, educação e renda de Castanhal em abas dinâmicas no Streamlit, para identificar padrões e disparidades na população.

**Why this priority**: É a funcionalidade core do projeto — sem visualização dos dados, nenhum outro módulo tem contexto.

**Independent Test**: Acesso à aba de visualização mostrando ao menos um gráfico correto para cada categoria de dado.

**Acceptance Scenarios**:

1. **Given** o app Streamlit está aberto, **When** seleciono a aba "Demografia", **Then** vejo gráficos de distribuição etária, pirâmide etária e distribuição étnico-racial de Castanhal.
2. **Given** o app está aberto, **When** seleciono a aba "Renda", **Then** vejo distribuição de renda per capita e Índice de Gini (quando disponível).

---

### User Story 2 — Resultados de ML (Classificação e Regressão) (Priority: P2)

Como analista, quero visualizar no Streamlit os resultados dos modelos de classificação e regressão treinados no Colab, para interpretar padrões previstos pela IA sobre a população de Castanhal.

**Why this priority**: Os modelos ML são o diferencial analítico do TCC.

**Independent Test**: Aba de ML exibe métricas do modelo (acurácia, R², etc.) e visualizações dos resultados (matriz de confusão, gráfico de regressão).

**Acceptance Scenarios**:

1. **Given** o pipeline foi executado no Colab e os resultados salvos, **When** acesso a aba "Machine Learning" no Streamlit, **Then** vejo métricas do modelo e gráficos interpretáveis.
2. **Given** resultados de classificação disponíveis, **When** interajo com filtros (ex: faixa etária), **Then** os gráficos atualizam para o subgrupo selecionado.

---

### User Story 3 — Assistente de IA Conversacional sobre os Dados (Priority: P2)

Como usuário do sistema, quero fazer perguntas em linguagem natural sobre os dados do censo e receber respostas contextualizadas geradas por IA, para entender os dados sem precisar de conhecimento técnico.

**Why this priority**: Agrega acessibilidade e diferencial acadêmico ao projeto.

**Independent Test**: Campo de pergunta funcional que retorna resposta coerente sobre ao menos uma variável do censo.

**Acceptance Scenarios**:

1. **Given** estou na aba "Perguntar à IA", **When** digito "Qual é a taxa de analfabetismo em Castanhal?", **Then** recebo uma resposta contextualizada com os dados disponíveis.
2. **Given** o assistente está ativo, **When** faço perguntas em sequência, **Then** o histórico da conversa fica visível na tela (dinamismo conversacional).

---

### User Story 4 — Aba de Políticas Públicas (Priority: P3)

Como gestor público ou pesquisador, quero visualizar as políticas públicas municipais de Castanhal e ver como os resultados de ML (classificação/regressão) podem ser aplicados para melhorá-las, para embasar tomadas de decisão.

**Why this priority**: Finalidade aplicada do TCC — conectar dados a ações concretas.

**Independent Test**: Aba "Políticas Públicas" exibe ao menos 3 políticas com sugestões de uso dos dados ML.

**Acceptance Scenarios**:

1. **Given** estou na aba "Políticas Públicas", **When** seleciono uma política (ex: educação), **Then** vejo uma análise de como os dados de escolaridade e os resultados do modelo podem orientar intervenções.

---

### User Story 5 — Pipeline Integrado Colab → GitHub → Streamlit (Priority: P1)

Como desenvolvedor do TCC, quero um pipeline documentado e funcional que puxe os dados do Google Drive, execute limpeza e treinamento no Colab e persista os artefatos (modelos, dados limpos, resultados) no GitHub para consumo no Streamlit.

**Why this priority**: Infraestrutura que sustenta todo o projeto.

**Independent Test**: Execução do notebook Colab gera arquivos no GitHub que o Streamlit lê corretamente.

**Acceptance Scenarios**:

1. **Given** os XLSX do Censo já estão no repositório GitHub em `censo_castanhal/censo_castanhal/`, **When** executo o notebook Colab, **Then** dados limpos e resultados ML são salvos no repositório GitHub em `data/processed/`, `models/` e `data/results/`.
2. **Given** artefatos atualizados no GitHub, **When** o Streamlit é (re)iniciado, **Then** lê os dados mais recentes automaticamente.

---

### Edge Cases

- O que acontece quando o arquivo XLSX do Drive está ausente ou corrompido no Colab?
- Como o Streamlit se comporta quando os artefatos de ML ainda não foram gerados?
- O que ocorre quando a IA não tem contexto suficiente para responder uma pergunta?
- Como lidar com dados ausentes (NaN) nas variáveis de renda ou escolaridade?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: O pipeline DEVE ler os dados XLSX do Censo 2022 diretamente do **repositório GitHub** (`censo_castanhal/censo_castanhal/`) via URL raw, processar no Colab e salvar artefatos limpos de volta no GitHub. Os dados XLSX do IBGE são públicos e estão commitados no repositório. O disparo do pipeline é **manual** — o desenvolvedor executa o notebook Colab quando precisar atualizar dados ou re-treinar modelos.
- **FR-002**: O Colab DEVE treinar modelos de classificação e regressão sobre as variáveis censitárias e salvar os artefatos (modelos serializados + resultados) no GitHub.
- **FR-003**: O Streamlit DEVE exibir abas dinâmicas para: Exploração de Dados, Machine Learning, IA Conversacional e Políticas Públicas.
- **FR-004**: O assistente de IA DEVE gerar respostas contextualizadas sobre os dados do censo em linguagem natural com histórico conversacional visível.
- **FR-005**: A aba de Políticas Públicas DEVE apresentar políticas municipais de Castanhal e como os resultados de ML podem ser aplicados a cada uma.
- **FR-006**: O Streamlit DEVE ler artefatos (modelos, dados processados) diretamente do repositório GitHub e será implantado no **Streamlit Community Cloud**, com URL pública para acesso da banca avaliadora. O app é **público, sem autenticação**.
- **FR-007**: O sistema DEVE usar **Google Gemini** (via `google-generativeai` SDK) para gerar respostas do assistente conversacional, aproveitando a integração nativa com o ecossistema Google (Colab, Drive).
- **FR-008**: O modelo de **classificação** DEVE prever **nível de vulnerabilidade socioeconômica** (ex: baixo/médio/alto) combinando variáveis de renda, escolaridade e condições de moradia. O modelo de **regressão** DEVE estimar indicadores de **infraestrutura urbana** (ex: acesso a saneamento, energia, coleta de lixo por setor censitário).

### Key Entities

- **CensoDataset**: Dados brutos e limpos do Censo 2022 de Castanhal (demografias, domicílios, educação, trabalho/renda)
- **MLModel**: Modelos de classificação e regressão treinados no Colab e serializados
- **MLResult**: Métricas e predições geradas pelos modelos (para exibição no Streamlit)
- **PolicyRecommendation**: Políticas públicas de Castanhal com análises de aplicação dos dados ML
- **ChatSession**: Histórico de conversa do usuário com o assistente de IA

## Clarifications

### Session 2026-03-16

- Q: Qual provedor de LLM será usado para o assistente de IA conversacional? → A: Google Gemini (google-generativeai SDK)
- Q: Quais são as variáveis-alvo dos modelos de ML (classificação e regressão)? → A: Vulnerabilidade socioeconômica (classificação) e infraestrutura urbana (regressão)
- Q: Onde o Streamlit será implantado? → A: Streamlit Community Cloud (deploy direto do GitHub, URL pública gratuita)
- Q: Como o pipeline de processamento será disparado? → A: Manual — execução do notebook Colab quando necessário
- Q: O app Streamlit exige autenticação/login? → A: Público, sem login — qualquer pessoa com a URL acessa

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Pipeline Colab executa end-to-end (GitHub XLSX → limpeza → treinamento → push de artefatos para GitHub) com exit code 0 e arquivos `.parquet`, `.joblib` e `.json` visíveis no repositório após a execução.
- **SC-002**: Streamlit exibe visualizações para todas as 4 categorias de dados (demografia, domicílios, educação, renda).
- **SC-003**: Assistente de IA responde corretamente a ao menos 80% das perguntas sobre os dados em testes manuais.
- **SC-004**: Aba de Políticas Públicas conecta ao menos 3 políticas municipais com recomendações baseadas nos resultados de ML.
- **SC-005**: Dados e artefatos persistidos no GitHub garantem que nenhum arquivo seja perdido entre sessões do Colab.
