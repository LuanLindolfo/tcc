import io

import joblib
import pandas as pd
import requests
import streamlit as st

from utils.config import get_github_raw_base


def _raw_url(path: str) -> str:
    return f"{get_github_raw_base()}/{path}"


@st.cache_resource
def load_model(nome: str):
    """
    Carrega modelo .joblib do GitHub.
    nome: ex. 'random_forest_ivs', 'xgboost_iah', 'kmeans_ocupacao'
    """
    url = _raw_url(f"models/{nome}.joblib")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return joblib.load(io.BytesIO(response.content))
    except Exception as e:
        st.warning(
            f"⚠️ Modelo `{nome}` não encontrado no GitHub. "
            "Execute o notebook Colab para gerar os artefatos de ML.\n\n"
            f"Detalhe: {e}"
        )
        return None


def predict_ivs(df: pd.DataFrame, modelo) -> pd.Series | None:
    """Classifica vulnerabilidade socioeconômica (baixa/media/alta)."""
    if modelo is None or df.empty:
        return None
    try:
        feature_cols = [
            c for c in df.columns
            if c not in ("setor_id", "iah", "ivs_label", "ivs_cluster", "zona_envelhecimento")
        ]
        return pd.Series(modelo.predict(df[feature_cols]), index=df.index)
    except Exception as e:
        st.error(f"Erro na predição IVS: {e}")
        return None


def predict_iah(df: pd.DataFrame, modelo) -> pd.Series | None:
    """Regride o Índice de Adequação Habitacional."""
    if modelo is None or df.empty:
        return None
    try:
        feature_cols = [
            c for c in df.columns
            if c not in ("setor_id", "iah", "ivs_label", "ivs_cluster", "zona_envelhecimento")
        ]
        return pd.Series(modelo.predict(df[feature_cols]), index=df.index)
    except Exception as e:
        st.error(f"Erro na predição IAH: {e}")
        return None
