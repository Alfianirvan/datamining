"""Pemuatan dan validasi data."""
from pathlib import Path

import pandas as pd
import streamlit as st

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "online_course_engagement_data.csv"
TARGET = "CourseCompletion"
ID_COL = "UserID"
LEAK_COL = "CompletionRate"
CAT_COL = "CourseCategory"
BINARY_COL = "DeviceType"
NUMERIC_COLS = [
    "TimeSpentOnCourse",
    "NumberOfVideosWatched",
    "NumberOfQuizzesTaken",
    "QuizScores",
    "CompletionRate",
]
REQUIRED_COLS = [ID_COL, CAT_COL, *NUMERIC_COLS, BINARY_COL, TARGET]


class DataError(ValueError):
    """Error validasi dataset."""


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Pastikan kolom wajib ada, bertipe benar, dan target biner."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise DataError(f"Kolom wajib tidak ditemukan: {', '.join(missing)}")
    for col in NUMERIC_COLS + [BINARY_COL, TARGET]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataError(f"Kolom '{col}' harus numerik.")
    if df[TARGET].isna().any():
        raise DataError(f"Kolom target '{TARGET}' memiliki nilai kosong.")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise DataError(f"Kolom target '{TARGET}' harus bernilai 0/1.")
    if len(df) < 20:
        raise DataError("Dataset terlalu kecil (minimal 20 baris).")
    return df[REQUIRED_COLS].copy()


@st.cache_data(show_spinner=False)
def load_data(uploaded_bytes: bytes | None = None) -> pd.DataFrame:
    """Muat CSV default, atau CSV hasil upload bila ada (dengan validasi)."""
    if uploaded_bytes is None:
        df = pd.read_csv(DEFAULT_PATH)
    else:
        import io

        try:
            df = pd.read_csv(io.BytesIO(uploaded_bytes))
        except Exception as exc:  # format salah / bukan CSV
            raise DataError(f"File tidak dapat dibaca sebagai CSV: {exc}") from exc
    return validate(df)
