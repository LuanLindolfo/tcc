<!--
SYNC IMPACT REPORT
==================
Version change: [template / unreleased] → 1.0.0
Type of bump: MINOR — primeira ratificação a partir do template genérico;
              4 princípios concretos adicionados, seções de restrições e governança preenchidas.

Princípios adicionados (novos):
  - I.  LGPD & Anonimização por Agregação
  - II. IA Ética como Ferramenta de Apoio à Decisão
  - III. Segurança de Credenciais — Zero-Exposure
  - IV. Rigor Acadêmico e Persistência Estruturada (.parquet)

Seções adicionadas:
  - Restrições de Escopo & Itens Diferidos
  - Governança (procedimento de emenda, versionamento, compliance)

Templates verificados:
  ✅ .specify/templates/plan-template.md — Constitution Check gates compatíveis
  ✅ .specify/templates/spec-template.md — Seção Requirements alinhada com FR existentes
  ✅ .specify/templates/tasks-template.md — Sem referências a princípios antigos
  ✅ specs/001-censo-streamlit-dashboard/plan.md — Constitution Check retroativamente alinhado

Itens diferidos (TODOs):
  - TODO(GINI_IMPL): Índice Gini presente no schema (trabalho_renda.parquet campo indice_gini)
    mas cálculo e exibição DIFERIDOS. Implementar quando o autor confirmar disponibilidade
    dos microdados necessários. Ver Princípio IV e seção de Restrições.

Follow-up necessário:
  - Após implementação do Gini: emenda PATCH atualizando seção de Restrições para
    remover o item diferido e adicionar critério de validação ao SC-002.
-->

# Constituição — Sistema de Inteligência Territorial de Castanhal

## Princípios Fundamentais

### I. LGPD & Anonimização por Agregação (NON-NEGOTIABLE)

Todo processamento de dados demográficos, habitacionais, educacionais e de renda do
Censo de Castanhal DEVE ocorrer exclusivamente sobre **dados agregados por setor
censitário ou município** — nunca sobre registros individuais ou domicílios específicos.

Regras não-negociáveis:
- Os dados brutos (XLSX do IBGE) DEVEM permanecer exclusivamente no Google Drive do
  autor; NUNCA devem ser commitados no repositório GitHub.
- Qualquer variável derivada ou composta (IAH, IVS, IDE) DEVE ser calculada sobre
  estatísticas agregadas, não sobre microdados individuais.
- Indicadores de saneamento, escolaridade e renda DEVEM ser utilizados estritamente
  para finalidades públicas e acadêmicas; uso comercial ou de identificação pessoal
  é vedado.
- O repositório GitHub DEVE conter apenas artefatos processados e anônimos (`.parquet`
  com médias/percentuais por setor, modelos `.joblib`, métricas `.json`).
- Nenhum campo de identificação de pessoa natural (CPF, nome, endereço) pode existir
  em qualquer arquivo do projeto — o Censo IBGE por design não os contém, mas
  qualquer enriquecimento futuro de dados DEVE respeitar este princípio.

**Rationale**: Conformidade com a Lei Geral de Proteção de Dados (LGPD — Lei 13.709/2018),
Art. 7º (bases legais) e Art. 11º (dados sensíveis). A finalidade pública e acadêmica
provê base legal, desde que mantida a anonimização efetiva.

---

### II. IA Ética como Ferramenta de Apoio à Decisão

O componente de Inteligência Artificial (modelos ML + assistente Gemini) DEVE operar
estritamente como **ferramenta de apoio à decisão**, não como substituta de análise
humana ou tomadora de decisão autônoma.

Regras não-negociáveis:
- Os modelos de classificação e regressão DEVEM ter seus **pesos e feature importances
  disponibilizados publicamente** no repositório GitHub — transparência total.
- Toda visualização de resultado de ML DEVE acompanhar as **métricas de avaliação**
  (acurácia, F1, R², RMSE) para que o usuário possa julgar a confiabilidade.
- O assistente Gemini DEVE sempre indicar que suas respostas são baseadas nos dados
  disponíveis e PODEM conter imprecisões — sem apresentar conclusões como definitivas.
- Análises de grupos étnico-raciais, fluxos migratórios e distribuição de renda DEVEM
  incluir advertência explícita sobre limitações do modelo e possíveis vieses nos
  dados fonte.
- Nenhum resultado de ML DEVE ser apresentado como "diagnóstico oficial" ou substituto
  de políticas públicas já estabelecidas.

**Rationale**: Modelos treinados em dados censitários históricos podem perpetuar ou
amplificar disparidades existentes. A transparência algorítmica e o framing correto
(apoio à decisão, não determinismo) são requisitos éticos e acadêmicos para um TCC
que influencia o debate sobre políticas públicas em Castanhal.

---

### III. Segurança de Credenciais — Zero-Exposure

Nenhuma chave de API, token de acesso ou credencial de serviço DEVE, em nenhuma
circunstância, aparecer em código-fonte, outputs de células, logs, commits ou
arquivos não protegidos do projeto.

Regras não-negociáveis:
- `GITHUB_TOKEN` DEVE ser armazenado exclusivamente em **Colab Secrets**
  (`google.colab.userdata`) — nunca em variáveis de célula visíveis ou comentários.
- `GEMINI_API_KEY` DEVE ser armazenada exclusivamente em **Streamlit Secrets**
  (`st.secrets`) para produção e em arquivo `.env` local (bloqueado por `.gitignore`)
  para desenvolvimento.
- O arquivo `.streamlit/secrets.toml` DEVE constar no `.gitignore` e NUNCA ser
  commitado; apenas o `.streamlit/secrets.toml.example` (sem valores) é permitido.
- O pipeline Colab→GitHub→Streamlit DEVE ser verificável sem expor nenhum secret:
  qualquer pessoa pode inspecionar o código sem comprometer as credenciais do autor.
- Em caso de exposição acidental (commit com token), o token DEVE ser revogado
  imediatamente e um novo gerado antes de continuar qualquer trabalho.

**Rationale**: A natureza pública do repositório GitHub torna a exposição de credenciais
um risco direto e imediato. A cadeia Drive→Colab→GitHub→Streamlit envolve múltiplos
serviços com credenciais distintas — disciplina rigorosa é obrigatória.

---

### IV. Rigor Acadêmico e Persistência Estruturada

Todo dado processado DEVE ser persistido em formato `.parquet` e todo código DEVE
atender ao padrão de reprodutibilidade exigido por um TCC de alta qualidade acadêmica.

Regras não-negociáveis:
- Dados processados DEVEM ser salvos em `.parquet` (via `pyarrow`) — NUNCA em `.csv`,
  `.pickle` ou outros formatos não tipados ou inseguros.
- Modelos ML DEVEM ser serializados em `.joblib` com `random_state` fixo (42) para
  garantir reprodutibilidade dos experimentos.
- O notebook Colab DEVE ser estruturado de forma que qualquer pesquisador possa
  re-executar o pipeline completo a partir dos XLSX originais e obter resultados
  idênticos (determinismo).
- Todas as funções de carga de dados no Streamlit DEVEM usar `@st.cache_data(ttl=3600)`
  para desempenho e consistência de exibição.
- O `GITHUB_RAW_BASE` DEVE ser configurável via `st.secrets` — nunca hardcoded —
  permitindo que o projeto seja bifurcado e reutilizado por outros pesquisadores.
- O campo `indice_gini` existe no schema de `trabalho_renda.parquet` mas seu **cálculo
  e exibição estão DIFERIDOS** até confirmação da disponibilidade dos microdados
  necessários. Ver TODO(GINI_IMPL) acima.

**Rationale**: Formato `.parquet` preserva tipos nativos, reduz tamanho dos artefatos
e acelera leitura no Streamlit. Reprodutibilidade é critério de avaliação acadêmica
explícito em bancas de TCC — um pipeline não-reprodutível compromete a validade
científica do trabalho.

---

## Restrições de Escopo & Itens Diferidos

Esta seção registra decisões de escopo e itens intencionalmente excluídos da versão
atual do projeto.

| Item | Status | Condição para ativação |
|------|--------|----------------------|
| **Índice de Gini** | ⏳ DIFERIDO | Implementar quando microdados de renda por setor estiverem disponíveis e validados. Campo `indice_gini` reservado no schema. |
| **Comparação histórica Censo 2010** | ⏳ DIFERIDO | Integrar se os XLSX do Censo 2010 forem incluídos no Drive. Nenhuma tarefa ativa. |
| **Autenticação de usuário** | ❌ DESCARTADO | App público por decisão de clarificação — banca deve acessar sem fricção. |
| **Pipeline automatizado (GitHub Actions)** | ❌ DESCARTADO | Dados do Censo 2022 são estáticos; automação é overkill para o escopo do TCC. |
| **Vector store / RAG completo** | ❌ DESCARTADO | Context injection via Gemini 1.5 Flash (128k tokens) é suficiente para o volume de dados. |

---

## Governança

**Autoridade**: Esta constituição tem precedência sobre todos os outros documentos
de planejamento (`spec.md`, `plan.md`, `tasks.md`, `research.md`). Em caso de
conflito, a constituição prevalece e os demais documentos DEVEM ser emendados.

**Procedimento de emenda**:
1. Identificar o princípio ou seção a ser alterado e justificar a necessidade.
2. Documentar a alteração no Sync Impact Report (comentário HTML no topo do arquivo).
3. Incrementar a versão seguindo semântica:
   - `MAJOR` (x.0.0): Remoção ou redefinição incompatível de um princípio existente.
   - `MINOR` (x.y.0): Adição de novo princípio ou seção, ou expansão material de escopo.
   - `PATCH` (x.y.z): Clarificações de linguagem, correções de typo, ajustes não-semânticos.
4. Atualizar `LAST_AMENDED_DATE` para a data atual.
5. Verificar e atualizar todos os templates e artefatos dependentes listados no Sync Impact Report.

**Compliance**:
- Todo novo requisito funcional adicionado ao `spec.md` DEVE ser verificado contra
  os 4 princípios desta constituição antes de ser aceito.
- Todo plano de implementação (`plan.md`) DEVE incluir uma seção "Constitution Check"
  que avalie explicitamente cada princípio.
- Qualquer desvio de princípio DEVE ser justificado na seção "Complexity Tracking"
  do `plan.md` com alternativa considerada e motivo de rejeição.
- A revisão de compliance DEVE ocorrer: (a) antes de iniciar a implementação de cada
  fase; (b) ao submeter o TCC para avaliação da banca.

**Version**: 1.0.0 | **Ratified**: 2026-03-16 | **Last Amended**: 2026-03-16
