# Contrato de Limpeza de Tabelas IBGE (Excel com Metadados)

**Projeto**: Sistema de Inteligência Territorial de Castanhal  
**Data**: 2026-03-23  
**Notebook**: `tcc_tabelas_merge.ipynb` — tabelas consolidadas por tópico do documento TCC

**Período dos dados**: 2010–2022 (Censo Demográfico IBGE). Dados anteriores a 2010 não são utilizados.

---

## Contexto

Muitos arquivos XLSX do IBGE (CNAE, taxa de atividade, profissões, distribuição de renda) possuem **metadados no topo** do arquivo. O header real dos dados fica em linhas variáveis. Este contrato documenta o padrão de limpeza usado para extrair dados tabulares consistentes.

---

## Padrão Geral de Limpeza

1. **Ler com `header=None`** — evita que pandas interprete linhas de metadados como cabeçalho
2. **Localizar dados por texto** — buscar string-chave na coluna de índice (ex: "Economicamente ativas", "Ano x Grandes grupos...")
3. **Extrair colunas** — mapear `iloc[row, col]` para campos desejados
4. **Remover metadados** — linhas vazias, NaN, "Total"
5. **`reset_index(drop=True)`** — índice limpo no output

---

## Funções Obrigatórias

### 1. `limpar_taxa_atividade(df_raw)` → `pd.DataFrame`

**Arquivo fonte**: `taxa_atividade.xlsx` (ano: 2022)  
**Saída**: DataFrame com `Condicao_Atividade`, `Total`, `Codigo_Municipio`

| Lógica | Descrição |
|--------|-----------|
| Busca | Linha onde coluna 1 contém "Economicamente ativas" |
| Extração | 4 linhas seguintes: colunas 1 e 2 |
| Conversão | `Total` → `pd.to_numeric` + `fillna(0).astype(int)` |
| Fixo | `Codigo_Municipio = 1502400` (Castanhal) |

---

### 2. `limpar_taxa_atividade_percentual(_)` → `pd.DataFrame`

**Arquivo fonte**: `taxa_atividade_percentual.xlsx` (ano: 2022)  
**Saída**: DataFrame com `Seção de Atividade`, `Valor`

| Lógica | Descrição |
|--------|-----------|
| Busca | Linha onde coluna 1 contém "Ano x Seção de atividade do trabalho principal" |
| Header | Nomes das colunas em `header_idx + 2` |
| Valores | Valores em `header_idx + 3` |
| Exclusões | Linhas vazias, NaN, "Total" em `Seção de Atividade` |
| Conversão | `Valor` → `pd.to_numeric(str(v).replace(',', '.'))` |

---

### 3. `limpar_profissoes(_)` → `pd.DataFrame`

**Arquivo fonte**: `profissões.xlsx` (ano: 2022)  
**Saída**: DataFrame com `Grupo de Ocupação`, `Valor`

| Lógica | Descrição |
|--------|-----------|
| Busca | Linha onde coluna 1 contém "Ano x Grandes grupos de ocupação no trabalho principal" |
| Header | Nomes em `header_idx + 2`, colunas 2+ |
| Valores | Valores em `header_idx + 3` |
| Exclusões | `dropna(subset=['Grupo de Ocupação'])`, excluir vazios, "nan", "Total" |
| Conversão | `Valor` → int |

---

### 4. `limpar_distribuicao_renda(_)` → `pd.DataFrame`

**Arquivo fonte**: `distribuição de renda.xlsx` (ano: 2022)  
**Saída**: DataFrame com `Classes_de_Rendimento`, `Total`

| Lógica | Descrição |
|--------|-----------|
| Busca | Linha onde coluna 1 contém "Até 1/4 de salário mínimo" |
| Extração | Linhas a partir de `start`, colunas 1 e 2 |
| Exclusões | NaN, vazios, linhas com "Total" em `Classes_de_Rendimento` |
| Conversão | `Total` → int |

---

### 5. `extrair_esc(df_raw, sexo, pct=False)` → `pd.DataFrame`

**Arquivos fonte** (ano: 2022): `analise_escolaridade_mulheres.xlsx`, `analise_escolaridade_homens.xlsx`, `analise_escolaridade_mulheres_percetual.xlsx`, `analise_escolaridade_homens_percetual.xlsx`  
**Saída**: DataFrame com `Municipio`, `Sexo`, `Ano`, e colunas de escolaridade

| Lógica | Descrição |
|--------|-----------|
| Busca | Linha onde **coluna 1** é igual a `sexo` (`'Mulheres'` ou `'Homens'`) — estrutura real: col 1 = Sexo |
| Campos | Colunas 3–7: Total_Pessoas_18_ou_mais, Sem_instr_fund_incompl, Fund_compl_medio_incompl, Medio_compl_super_incompl, Superior_completo |
| Variante | Se `pct=True`: sufixo `_Perc` nos nomes das colunas |

---

## Fontes de Dados

| Ambiente | Base | Fallback |
|----------|------|----------|
| **Colab** | `/content/drive/MyDrive/censo_castanhal/` | — |
| **Local** | `censo_castanhal/` | `https://raw.githubusercontent.com/LuanLindolfo/tcc/main/censo_castanhal/censo_castanhal/` |

Função auxiliar `_get_source(path)`:
- Se arquivo existe localmente → retorna path
- Se `USAR_GITHUB` e não encontrado → baixa via `requests.get(GITHUB_RAW/.../path)`

---

## Invariantes Pós-Limpeza

- Nenhuma linha com `Classes_de_Rendimento` ou `Seção de Atividade` ou `Grupo de Ocupação` igual a `""`, `"nan"` ou `"Total"`
- Colunas numéricas sem NaN (ou `fillna(0)` conforme aplicável)
- `reset_index(drop=True)` em todos os DataFrames retornados
