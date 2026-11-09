"""Feature engineering dan preprocessor (semua di dalam Pipeline)."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from src.data import BINARY_COL, CAT_COL, ID_COL, LEAK_COL, NUMERIC_COLS, TARGET

SCENARIO_A = "A: Early warning (drop CompletionRate)"
SCENARIO_B = "B: Progress tracking (keep CompletionRate)"


def build_features(df: pd.DataFrame, scenario: str):
    """Pisahkan X dan y; drop UserID, dan CompletionRate pada skenario A."""
    drop = [TARGET, ID_COL]
    if scenario == SCENARIO_A:
        drop.append(LEAK_COL)
    X = df.drop(columns=drop)
    y = df[TARGET].astype(int)
    return X, y


def numeric_features(X: pd.DataFrame) -> list[str]:
    """Kolom numerik kontinu/hitung yang akan di-scale."""
    return [c for c in NUMERIC_COLS if c in X.columns]


def build_preprocessor(X: pd.DataFrame, scaler_name: str) -> ColumnTransformer:
    """ColumnTransformer: one-hot CourseCategory, scaling numerik, DeviceType passthrough."""
    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "None": "passthrough",
    }
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [CAT_COL]),
            ("num", scalers[scaler_name], numeric_features(X)),
            ("bin", "passthrough", [BINARY_COL]),
        ]
    )


def split_data(X, y, test_size: float, seed: int):
    """Stratified holdout split."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
