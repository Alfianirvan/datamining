"""Definisi model dan proses training/evaluasi."""
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessor

ALL_MODELS = [
    "Naive Bayes",
    "k-Nearest Neighbors",
    "Decision Tree",
    "Logistic Regression",
    "Random Forest",
]
DEFAULT_MODELS = ALL_MODELS[:3]


def get_models(selected, k: int, max_depth, balanced: bool, seed: int) -> dict:
    """Bangun dict nama -> estimator sesuai hyperparameter sidebar."""
    cw = "balanced" if balanced else None
    catalog = {
        "Naive Bayes": lambda: GaussianNB(),  # tidak mendukung class_weight
        "k-Nearest Neighbors": lambda: KNeighborsClassifier(n_neighbors=k),  # idem
        "Decision Tree": lambda: DecisionTreeClassifier(
            max_depth=max_depth, class_weight=cw, random_state=seed
        ),
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=1000, class_weight=cw, random_state=seed
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=200, class_weight=cw, random_state=seed, n_jobs=-1
        ),
    }
    return {name: catalog[name]() for name in selected}


def train_and_evaluate(models, X_train, X_test, y_train, y_test, scaler_name, cv_folds, seed):
    """Latih tiap model dalam Pipeline, hitung metrik holdout + CV (kelas positif = 0)."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    results = {}
    for name, est in models.items():
        pipe = Pipeline(
            [("prep", build_preprocessor(X_train, scaler_name)), ("clf", clone(est))]
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        # skor untuk kelas 0 (Not Completed) = positif untuk early warning
        proba0 = pipe.predict_proba(X_test)[:, list(pipe.classes_).index(0)]
        cv_scores = cross_val_score(
            Pipeline([("prep", build_preprocessor(X_train, scaler_name)), ("clf", clone(est))]),
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
        )
        results[name] = {
            "pipeline": pipe,
            "y_pred": y_pred,
            "proba0": proba0,
            "metrics": {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision (Not Completed)": precision_score(y_test, y_pred, pos_label=0, zero_division=0),
                "Recall (Not Completed)": recall_score(y_test, y_pred, pos_label=0, zero_division=0),
                "Macro F1": f1_score(y_test, y_pred, average="macro"),
                "ROC-AUC": roc_auc_score((np.asarray(y_test) == 0).astype(int), proba0),
                "CV Accuracy mean": cv_scores.mean(),
                "CV Accuracy std": cv_scores.std(),
            },
        }
    return results
