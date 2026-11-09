"""Metrik, confusion matrix, ROC, dan feature importance."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

METRIC_COLS = [
    "Accuracy",
    "Precision (Not Completed)",
    "Recall (Not Completed)",
    "Macro F1",
    "ROC-AUC",
]


def leaderboard(results: dict) -> pd.DataFrame:
    """Tabel metrik semua model (numerik) + kolom CV tampilan."""
    rows = {}
    for name, r in results.items():
        m = r["metrics"]
        row = {c: m[c] for c in METRIC_COLS}
        row["CV Accuracy (mean ± std)"] = f"{m['CV Accuracy mean']:.4f} ± {m['CV Accuracy std']:.4f}"
        row["_cv_mean"] = m["CV Accuracy mean"]
        rows[name] = row
    return pd.DataFrame(rows).T.astype({**{c: float for c in METRIC_COLS}, "_cv_mean": float})


def confusion_parts(y_true, y_pred):
    """Confusion matrix dengan kelas positif = 0 (Not Completed): TP, FP, FN, TN."""
    yt = (np.asarray(y_true) == 0).astype(int)
    yp = (np.asarray(y_pred) == 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return int(tp), int(fp), int(fn), int(tn)


def roc_points(y_true, proba0):
    """Titik ROC untuk kelas positif 0."""
    return roc_curve((np.asarray(y_true) == 0).astype(int), proba0)


def report_frame(y_true, y_pred) -> pd.DataFrame:
    """Classification report sebagai DataFrame."""
    rep = classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=["Not Completed (0)", "Completed (1)"],
        output_dict=True, zero_division=0,
    )
    return pd.DataFrame(rep).T


def feature_importance(pipe, X_test, y_test, seed: int) -> pd.DataFrame:
    """Importance bawaan (tree) bila ada, jika tidak permutation importance (pada fitur asli)."""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        names = pipe.named_steps["prep"].get_feature_names_out()
        names = [n.split("__", 1)[-1] for n in names]
        return pd.DataFrame({"feature": names, "importance": clf.feature_importances_, "method": "impurity"}) \
            .sort_values("importance", ascending=False)
    res = permutation_importance(
        pipe, X_test, y_test, n_repeats=5, random_state=seed, scoring="accuracy"
    )
    return pd.DataFrame(
        {"feature": X_test.columns, "importance": res.importances_mean, "method": "permutation"}
    ).sort_values("importance", ascending=False)
