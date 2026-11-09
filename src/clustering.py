"""K-means, elbow, silhouette, dan penamaan segmen."""
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data import TARGET

ENGAGEMENT_COLS = ["TimeSpentOnCourse", "NumberOfVideosWatched", "NumberOfQuizzesTaken", "QuizScores"]


def scale_features(df: pd.DataFrame, cols):
    """Scale fitur engagement (tanpa target)."""
    return StandardScaler().fit_transform(df[cols])


@st.cache_data(show_spinner="Menghitung elbow dan silhouette...")
def elbow_table(Xs, seed: int, k_range=range(2, 11)) -> pd.DataFrame:
    """Inertia dan silhouette untuk tiap k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
        rows.append({"k": k, "SSE": km.inertia_, "Silhouette": silhouette_score(Xs, km.labels_)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def fit_kmeans(Xs, k: int, seed: int):
    """Fit k-means final dan proyeksi PCA 2D."""
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
    pca = PCA(n_components=2, random_state=seed)
    return km.labels_, pca.fit_transform(Xs), pca


def name_segments(profile: pd.DataFrame, cols) -> dict:
    """Beri nama segmen dari peringkat rata-rata z-score engagement (tinggi -> Aktif)."""
    z = (profile[cols] - profile[cols].mean()) / profile[cols].std(ddof=0).replace(0, 1)
    order = z.mean(axis=1).sort_values(ascending=False).index.tolist()
    k = len(order)
    labels = {2: ["Aktif", "Pasif"], 3: ["Aktif", "Sedang", "Pasif"]}.get(k)
    if labels is None:
        labels = ["Aktif"] + [f"Sedang-{i}" for i in range(1, k - 1)] + ["Pasif"]
    return {cl: f"{lab} (C{cl})" for cl, lab in zip(order, labels)}


def profile(df: pd.DataFrame, labels, cols) -> pd.DataFrame:
    """Rata-rata fitur, completion rate, dan ukuran per cluster."""
    d = df[cols + [TARGET]].copy()
    d["cluster"] = labels
    prof = d.groupby("cluster")[cols].mean()
    prof["Completion rate (%)"] = d.groupby("cluster")[TARGET].mean() * 100
    prof["Jumlah siswa"] = d.groupby("cluster").size()
    return prof
