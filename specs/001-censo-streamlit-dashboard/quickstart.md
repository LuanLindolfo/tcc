# Quickstart: Sistema de Inteligência Territorial de Castanhal

**Tempo estimado de setup**: 30–45 minutos (primeira vez)

---

## Pré-requisitos

| Ferramenta | Versão mínima | Onde obter |
|-----------|--------------|-----------|
| Python | 3.11+ | python.org |
| Git | qualquer | git-scm.com |
| Conta Google | — | google.com (para Colab + Drive) |
| Conta GitHub | — | github.com |
| Conta Streamlit Cloud | — | share.streamlit.io |

---

## Passo 1: Configurar o Repositório GitHub

```bash
# Clone o repositório (ou crie um novo)
git clone https://github.com/SEU_USUARIO/tcc-castanhal.git
cd tcc-castanhal

# Estrutura inicial de diretórios (sem pasta pages/ — app único em app.py)
mkdir -p data/processed data/results data/raw models utils
```

**Criar `.gitignore`**:
```gitignore
# Dados brutos (ficam apenas no Drive)
data/raw/
*.xlsx
*.xls

# Credenciais
.env
*.env
google_auth.json
credentials.json
service_account.json

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
*.egg-info/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Modelos temporários
*.tmp
```

**Criar `requirements.txt`**:
```txt
streamlit>=1.36.0
pandas>=2.0.0
pyarrow>=14.0.0
plotly>=5.18.0
scikit-learn>=1.4.0
xgboost>=2.0.0
joblib>=1.3.0
google-generativeai>=0.5.0
requests>=2.31.0
python-dotenv>=1.0.0
```

---

## Passo 2: Configurar Colab Secrets

1. Abrir o notebook no Google Colab: https://colab.research.google.com/drive/1DI1Xzzeo1JjgIgJQQr80zfOTLICUTG3p
2. No menu lateral, clicar em **🔑 Secrets**
3. Adicionar os seguintes secrets:

| Secret Name | Valor | Onde obter |
|------------|-------|-----------|
| `GITHUB_TOKEN` | `ghp_xxxxxxxxxxxx` | GitHub → Settings → Developer Settings → Personal Access Tokens → Fine-grained → repo write |

**Permissões mínimas do PAT**:
- `Contents: Read and Write`
- `Metadata: Read-only`

---

## Passo 3: Organizar os Dados no Google Drive

Estrutura esperada no Drive:
```
MyDrive/
└── TCC_Castanhal/
    └── dados/
        ├── censo2022_demografico.xlsx
        ├── censo2022_domicilios.xlsx
        ├── censo2022_educacao.xlsx
        └── censo2022_trabalho_renda.xlsx
```

> Os nomes exatos dos arquivos serão configurados em `DRIVE_BASE_PATH` no notebook.

---

## Passo 4: Executar o Pipeline no Colab

Execute as células na ordem:

```
[1] Setup & Autenticação  → Monta Drive, carrega GITHUB_TOKEN
[2] Ingestão              → Lê XLSX do Drive
[3] Limpeza               → Remove NaN, padroniza tipos
[4] Feature Engineering   → Calcula IAH, IDE, IVS
[5] EDA (opcional)        → Visualizações locais no Colab
[6] Treinamento RF         → Modelo de classificação IVS
[7] Treinamento XGBoost    → Modelo de regressão IAH
[8] Clustering K-Means     → Perfil de ocupação
[9] Exportação             → Salva .parquet, .joblib, .json
[10] Push GitHub           → Envia artefatos para o repositório
```

**Verificar sucesso**: Após a célula 10, os arquivos devem aparecer no GitHub em `data/processed/`, `data/results/` e `models/`.

---

## Passo 5: Configurar Streamlit Community Cloud

1. Acessar https://share.streamlit.io
2. Clicar em **New app**
3. Conectar ao repositório GitHub: `SEU_USUARIO/tcc-castanhal`
4. Branch: `main` | Main file path: `app.py`
5. Clicar em **Advanced settings** → **Secrets**
6. Adicionar:

```toml
GEMINI_API_KEY = "AIzaSy_sua_chave_aqui"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/SEU_USUARIO/tcc-castanhal/main"
```

**Onde obter a GEMINI_API_KEY**: https://aistudio.google.com/app/apikey (gratuito)

7. Clicar em **Deploy** → aguardar build (~2–3 min)

---

## Passo 6: Desenvolvimento Local (opcional)

```bash
# Instalar dependências
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou: .venv\Scripts\activate  # Windows

pip install -r requirements.txt

# Criar .env local (NÃO commitar!)
echo 'GEMINI_API_KEY="sua_chave"' > .env
echo 'GITHUB_RAW_BASE="https://raw.githubusercontent.com/SEU_USUARIO/tcc-castanhal/main"' >> .env

# Rodar o app
streamlit run app.py
```

---

## Verificação Rápida (Checklist)

- [ ] Repositório GitHub criado com `.gitignore` correto
- [ ] `GITHUB_TOKEN` configurado nos Colab Secrets
- [ ] Arquivos XLSX no Google Drive na estrutura correta
- [ ] Pipeline Colab executado sem erros
- [ ] Artefatos visíveis no GitHub (`data/processed/`, `models/`)
- [ ] `GEMINI_API_KEY` configurada nos Streamlit Secrets
- [ ] App rodando no Streamlit Community Cloud
- [ ] Todas as seções do menu (Início, Demografia, Domicílios, Educação & Renda, Políticas, Assistente IA) carregando sem erro

---

## Solução de Problemas Comuns

| Problema | Causa provável | Solução |
|---------|---------------|---------|
| `FileNotFoundError` no Colab | XLSX não encontrado no Drive | Verificar `DRIVE_BASE_PATH` e nome do arquivo |
| `401 Unauthorized` no push GitHub | Token expirado ou sem permissão | Gerar novo PAT com permissão `Contents: Write` |
| Streamlit mostra "Dados não disponíveis" | Pipeline não executado ou artefatos não no GitHub | Re-executar células 9–10 do notebook |
| Gemini retorna erro `403` | API key inválida ou cota excedida | Verificar chave em aistudio.google.com |
| `.parquet` não carrega no Streamlit | `pyarrow` não instalado | Adicionar `pyarrow` ao `requirements.txt` |
