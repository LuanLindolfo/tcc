"""Exportação de relatório simples em PDF (fpdf2) + CSV a partir das projeções."""

from __future__ import annotations

import io
import unicodedata
from datetime import datetime

import pandas as pd

from utils.censo_projecoes import SERIES, dataframe_relatorio_completo


def _ascii_safe(s: object) -> str:
    t = str(s)
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode() or "?"


def csv_bytes() -> bytes:
    df = dataframe_relatorio_completo()
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue().encode("utf-8-sig")


def pdf_bytes() -> bytes:
    """Gera PDF textual com resumo; requer fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError as e:
        raise ImportError("Instale fpdf2: pip install fpdf2") from e

    df = dataframe_relatorio_completo()
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, _ascii_safe("Castanhal-PA - Indicadores e previsoes (MLP)"), ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        0,
        5,
        _ascii_safe(
            "Relatorio gerado a partir do notebook Arquivo_tcc.ipynb (TCC). "
            "Previsoes com MLPRegressor (scikit-learn), fins academicos."
        ),
    )
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, _ascii_safe(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}"), ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 7)
    for _, row in df.iterrows():
        linha = " | ".join(_ascii_safe(f"{k}: {v}") for k, v in row.items() if pd.notna(v))
        pdf.multi_cell(0, 4, linha[:450])
        pdf.ln(1)
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(
        0,
        5,
        _ascii_safe("Aviso: nao substitui publicacoes oficiais do IBGE."),
    )
    out = pdf.output(dest="S")
    if isinstance(out, str):
        return out.encode("latin-1", errors="replace")
    return bytes(out)


def texto_apresentacao_series() -> str:
    """Parágrafo único para Home."""
    return (
        f"O notebook consolida {len(SERIES)} indicadores temáticos com histórico em um ou mais censos, "
        "incluindo projeções com redes neurais conforme a metodologia documentada no TCC."
    )
