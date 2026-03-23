import os
import streamlit as st

GITHUB_REPO = "LuanLindolfo/tcc"
GITHUB_BRANCH = "001-censo-streamlit-dashboard"
GITHUB_RAW_BASE_DEFAULT = (
    f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}"
)
XLSX_BASE_PATH = "censo_castanhal/censo_castanhal"


def get_github_raw_base() -> str:
    """Retorna URL base do repositório GitHub para leitura de artefatos."""
    try:
        return st.secrets["GITHUB_RAW_BASE"]
    except (KeyError, FileNotFoundError):
        return os.getenv("GITHUB_RAW_BASE", GITHUB_RAW_BASE_DEFAULT)
