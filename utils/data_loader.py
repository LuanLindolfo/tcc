import io
import re
import unicodedata

import pandas as pd
import requests
import streamlit as st

from utils.config import get_github_raw_base


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Retorna o primeiro nome de coluna encontrado no DataFrame dentre os candidatos.
    A busca é case-insensitive e ignora acentos.
    """
    def norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
        return re.sub(r"[^a-z0-9]", "_", s)

    normed_cols = {norm(c): c for c in df.columns}
    for candidate in candidates:
        key = norm(candidate)
        if key in normed_cols:
            return normed_cols[key]
        # busca parcial: o candidato está contido no nome da coluna
        for col_norm, col_orig in normed_cols.items():
            if key in col_norm:
                return col_orig
    return None


def col_sum(df: pd.DataFrame, candidates: list[str]) -> float | None:
    """Soma os valores de uma coluna encontrada via find_col. Retorna None se não encontrar."""
    col = find_col(df, candidates)
    return float(df[col].sum()) if col else None


def col_mean(df: pd.DataFrame, candidates: list[str]) -> float | None:
    """Média dos valores de uma coluna encontrada via find_col. Retorna None se não encontrar."""
    col = find_col(df, candidates)
    return float(df[col].mean()) if col else None


def _raw_url(path: str) -> str:
    return f"{get_github_raw_base()}/{path}"


def _load_parquet(path: str) -> pd.DataFrame:
    url = _raw_url(path)
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return pd.read_parquet(io.BytesIO(response.content))
    except Exception as e:
        st.error(
            f"❌ Não foi possível carregar `{path}` do GitHub. "
            "Execute o pipeline no Colab para gerar os dados.\n\n"
            f"Detalhe: {e}"
        )
        return pd.DataFrame()


def _load_json(path: str) -> list | dict:
    url = _raw_url(path)
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(
            f"❌ Não foi possível carregar `{path}` do GitHub.\n\nDetalhe: {e}"
        )
        return []


@st.cache_data(ttl=3600)
def load_demografico() -> pd.DataFrame:
    return _load_parquet("data/processed/demografico.parquet")


@st.cache_data(ttl=3600)
def load_domicilios() -> pd.DataFrame:
    return _load_parquet("data/processed/domicilios.parquet")


@st.cache_data(ttl=3600)
def load_educacao() -> pd.DataFrame:
    return _load_parquet("data/processed/educacao.parquet")


@st.cache_data(ttl=3600)
def load_trabalho_renda() -> pd.DataFrame:
    return _load_parquet("data/processed/trabalho_renda.parquet")


@st.cache_data(ttl=3600)
def load_features_compostas() -> pd.DataFrame:
    return _load_parquet("data/processed/features_compostas.parquet")


@st.cache_data(ttl=3600)
def load_politicas() -> list:
    return _load_json("data/results/politicas_recomendacoes.json")


@st.cache_data(ttl=3600)
def load_ml_metricas(tipo: str) -> dict:
    """tipo: 'classificacao' ou 'regressao'"""
    result = _load_json(f"data/results/ml_{tipo}_metricas.json")
    return result if isinstance(result, dict) else {}
