"""Online Course Engagement - Data Mining Testbed (Streamlit)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

from src import clustering as clu
from src import evaluation as ev
from src.data import (BINARY_COL, CAT_COL, ID_COL, LEAK_COL, NUMERIC_COLS, TARGET,
                      DataError, load_data)
from src.models import ALL_MODELS, DEFAULT_MODELS, get_models, train_and_evaluate
from src.preprocessing import (SCENARIO_A, SCENARIO_B, build_features,
                               build_preprocessor, split_data)

st.set_page_config(page_title="Course Engagement Testbed", page_icon="🎓", layout="wide")
FIGSIZE = (6, 4)
CLASS_NAMES = {0: "Not Completed (0)", 1: "Completed (1)"}
PALETTE = {0: "#d95f02", 1: "#1b9e77"}


def new_fig(size=FIGSIZE):
    """Buat figure dengan ukuran konsisten."""
    return plt.subplots(figsize=size)


def show(fig):
    """Tampilkan figure lalu tutup agar memori tidak bocor."""
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------ Sidebar
st.sidebar.title("Testbed Configuration")
upload = st.sidebar.file_uploader("Upload CSV (opsional)", type=["csv"])
try:
    df = load_data(upload.getvalue() if upload is not None else None)
except DataError as exc:
    st.error(f"Dataset tidak valid: {exc}")
    st.stop()
if upload is not None:
    st.sidebar.success(f"Memakai file upload: {upload.name}")

scenario = st.sidebar.radio("Leakage scenario", [SCENARIO_A, SCENARIO_B], index=0)
scaler_name = st.sidebar.selectbox("Feature scaler", ["StandardScaler", "MinMaxScaler", "None"])
test_size = st.sidebar.slider("Test size (holdout)", 0.10, 0.40, 0.20, 0.05)
cv_folds = st.sidebar.slider("Cross-validation folds (K)", 3, 10, 5)
selected = st.sidebar.multiselect("Models", ALL_MODELS, default=DEFAULT_MODELS)
with st.sidebar.expander("Hyperparameter"):
    knn_k = st.slider("k untuk kNN", 1, 25, 5)
    depth_none = st.checkbox("max_depth Decision Tree = None", value=False)
    depth_val = st.slider("max_depth Decision Tree", 1, 20, 5, disabled=depth_none)
    max_depth = None if depth_none else depth_val
balanced = st.sidebar.checkbox("Handle class imbalance (class_weight='balanced')")
seed = int(st.sidebar.number_input("Random seed", value=42, step=1))

config = dict(scenario=scenario, scaler=scaler_name, test_size=test_size, cv=cv_folds,
              models=tuple(selected), k=knn_k, depth=max_depth, balanced=balanced, seed=seed,
              data_sig=(len(df), float(df[TARGET].sum())))

# Basic data facts used across tabs
class_counts = df[TARGET].value_counts()
single_class = class_counts.shape[0] < 2
X, y = build_features(df, scenario)

st.title("🎓 Online Course Engagement — Data Mining Testbed")
tabs = st.tabs(["Overview", "EDA", "Preprocessing", "Modeling", "Evaluation", "Clustering"])

# ------------------------------------------------------------------ 1. Overview
with tabs[0]:
    st.caption("Fase CRISP-DM: Business Understanding & Data Understanding.")
    st.subheader("Tujuan Bisnis")
    st.write(
        "Memprediksi apakah siswa menyelesaikan kursus online (`CourseCompletion`) berdasarkan "
        "tingkat keterlibatan belajarnya, sehingga platform dapat mendeteksi siswa yang berisiko "
        "tidak menyelesaikan kursus sejak dini."
    )
    c = st.columns(5)
    c[0].metric("Rows", f"{len(df):,}")
    c[1].metric("Columns", df.shape[1])
    c[2].metric("Missing values", int(df.isna().sum().sum()))
    c[3].metric("Completion rate", f"{df[TARGET].mean() * 100:.1f}%")
    c[4].metric("Duplicate UserID", int(df[ID_COL].duplicated().sum()))
    st.subheader("Contoh Data (10 baris)")
    st.dataframe(df.head(10), width='stretch')
    st.subheader("Kamus Data")
    st.dataframe(pd.DataFrame({
        "Kolom": df.columns,
        "Tipe": [str(t) for t in df.dtypes],
        "Keterangan": [
            "ID siswa (bukan fitur, di-drop)",
            "Kategori kursus: Arts, Business, Health, Programming, Science",
            "Jam belajar pada kursus",
            "Jumlah video yang ditonton",
            "Jumlah kuis yang diikuti",
            "Rata-rata skor kuis",
            "Persentase progres kursus (kandidat data leakage)",
            "Tipe perangkat (asumsi: 0 = Desktop, 1 = Mobile)",
            "TARGET: 1 = menyelesaikan kursus, 0 = tidak",
        ],
    }), width='stretch', hide_index=True)
    st.info("Asumsi: `DeviceType` 0 = Desktop dan 1 = Mobile (dataset tidak mendokumentasikan label ini).")

# ------------------------------------------------------------------ 2. EDA
with tabs[1]:
    st.caption("Fase CRISP-DM: Data Understanding.")
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.drop(columns=[ID_COL]).describe().T, width='stretch')
    st.markdown("**Jenis atribut**")
    st.dataframe(pd.DataFrame({
        "Atribut": [ID_COL, CAT_COL, "TimeSpentOnCourse", "NumberOfVideosWatched",
                    "NumberOfQuizzesTaken", "QuizScores", LEAK_COL, BINARY_COL, TARGET],
        "Jenis": ["Identifier", "Nominal", "Rasio", "Rasio (cacah)", "Rasio (cacah)",
                  "Rasio", "Rasio", "Biner", "Biner (target)"],
    }), hide_index=True, width='stretch')

    st.subheader("Distribusi Target")
    fig, ax = new_fig()
    counts = df[TARGET].value_counts().sort_index()
    bars = ax.bar([CLASS_NAMES[i] for i in counts.index], counts.values,
                  color=[PALETTE[i] for i in counts.index])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v} ({v / len(df) * 100:.1f}%)",
                ha="center", va="bottom")
    ax.set(title="Distribusi CourseCompletion", xlabel="Kelas", ylabel="Jumlah siswa")
    show(fig)

    st.subheader("Variabel Numerik vs Target")
    var = st.selectbox("Pilih variabel numerik", NUMERIC_COLS, key="eda_num")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = new_fig()
        sns.histplot(data=df, x=var, hue=df[TARGET].map(CLASS_NAMES), bins=30, alpha=0.55,
                     ax=ax, palette={CLASS_NAMES[k]: v for k, v in PALETTE.items()})
        ax.set(title=f"Histogram {var} per kelas", xlabel=var, ylabel="Frekuensi")
        show(fig)
    with col2:
        fig, ax = new_fig()
        sns.boxplot(data=df, x=df[TARGET].map(CLASS_NAMES), y=var, ax=ax,
                    hue=df[TARGET].map(CLASS_NAMES), legend=False,
                    palette={CLASS_NAMES[k]: v for k, v in PALETTE.items()})
        ax.set(title=f"Boxplot {var} per kelas", xlabel="Kelas", ylabel=var)
        show(fig)
    g0, g1 = df.loc[df[TARGET] == 0, var], df.loc[df[TARGET] == 1, var]
    if len(g0) and len(g1):
        u, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        st.write(f"**Uji Mann-Whitney U**: U = {u:,.1f}, p-value = {p:.4g} → "
                 f"{'Signifikan' if p < 0.05 else 'Tidak signifikan'} pada α = 0,05")
        st.caption(f"Median kelas 0 = {g0.median():.2f}, median kelas 1 = {g1.median():.2f}")

    st.subheader("Variabel Kategorikal vs Target (Chi-square)")
    cat = st.selectbox("Pilih variabel kategorikal", [CAT_COL, BINARY_COL], key="eda_cat")
    ct = pd.crosstab(df[cat], df[TARGET].map(CLASS_NAMES))
    ct = ct[[CLASS_NAMES[k] for k in (0, 1) if CLASS_NAMES[k] in ct.columns]]
    st.markdown("**Tabel kontingensi**")
    st.dataframe(ct, width='stretch')
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    fig, ax = new_fig()
    pct.plot(kind="bar", stacked=True, ax=ax, rot=0,
             color=[PALETTE[0], PALETTE[1]][: pct.shape[1]] if pct.shape[1] == 2 else None)
    ax.set(title=f"Proporsi target per {cat}", xlabel=cat, ylabel="Persentase (%)")
    ax.legend(title="Kelas", loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    show(fig)
    if ct.shape[0] > 1 and ct.shape[1] > 1:
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        st.write(f"**Chi-square** = {chi2:.4f}, dof = {dof}, p-value = {p:.4g}")
        (st.error if p < 0.05 else st.success)(
            f"Kesimpulan: {'Signifikan' if p < 0.05 else 'Tidak signifikan'} pada α = 0,05 "
            f"({cat} {'berhubungan' if p < 0.05 else 'tidak berhubungan'} dengan CourseCompletion)."
        )
    else:
        st.warning("Uji chi-square memerlukan minimal 2 kategori dan 2 kelas target.")

    st.subheader("Heatmap Korelasi")
    corr = df[NUMERIC_COLS + [BINARY_COL, TARGET]].corr()
    fig, ax = new_fig((7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Korelasi antar variabel numerik")
    show(fig)

# ------------------------------------------------------------------ 3. Preprocessing
with tabs[2]:
    st.caption("Fase CRISP-DM: Data Preparation.")
    c = st.columns(4)
    c[0].metric("Skenario", scenario.split(":")[0])
    c[1].metric("Scaler", scaler_name)
    c[2].metric("Test size", f"{test_size:.2f}")
    c[3].metric("CV folds", cv_folds)
    st.markdown(f"""
**Langkah preprocessing**
1. **Drop `{ID_COL}`** — hanya pengenal siswa, tidak membawa informasi perilaku belajar; memakainya
   akan membuat model menghafal ID.
2. **`{LEAK_COL}`** — {"di-*drop* (skenario A, early warning: progres belum diketahui saat prediksi)" if scenario == SCENARIO_A else "dipertahankan (skenario B, progress tracking); skor tinggi berpotensi akibat leakage"}.
3. **One-hot encoding `{CAT_COL}`**.
4. **Scaling** fitur numerik dengan **{scaler_name}**.

Semua langkah berada di dalam `Pipeline` + `ColumnTransformer` dan di-*fit* hanya pada data latih.
""")
    if single_class:
        st.error("Target hanya memiliki satu kelas; split stratified dan modeling tidak dapat dilakukan.")
    else:
        try:
            X_train, X_test, y_train, y_test = split_data(X, y, test_size, seed)
        except ValueError as exc:
            st.error(f"Split gagal: {exc}")
            st.stop()
        prep = build_preprocessor(X_train, scaler_name).fit(X_train)
        Xt = prep.transform(X_train)
        names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
        st.success(f"Shape after encoding: {Xt.shape[0]} rows × {Xt.shape[1]} features")
        st.markdown("**Preview matriks fitur (data latih, setelah transformasi)**")
        st.dataframe(pd.DataFrame(Xt, columns=names).head(10), width='stretch')
        st.markdown("**Ukuran dan distribusi kelas (stratified)**")
        st.dataframe(pd.DataFrame({
            "Set": ["Train", "Test"],
            "Ukuran": [len(y_train), len(y_test)],
            "Not Completed (0)": [(y_train == 0).sum(), (y_test == 0).sum()],
            "Completed (1)": [(y_train == 1).sum(), (y_test == 1).sum()],
            "% Completed": [f"{y_train.mean() * 100:.1f}%", f"{y_test.mean() * 100:.1f}%"],
        }), hide_index=True, width='stretch')

# ------------------------------------------------------------------ 4. Modeling
with tabs[3]:
    st.caption("Fase CRISP-DM: Modeling.")
    st.write(f"Training on {len(selected)} selected model(s): "
             f"{', '.join(selected) if selected else '-'}")
    st.info("Kelas positif untuk early warning adalah **0 (Not Completed)**, sehingga Precision dan "
            "Recall dihitung untuk siswa yang tidak menyelesaikan kursus.")
    if not selected:
        st.warning("Pilih minimal satu model di sidebar.")
    if st.button("Train models", type="primary", disabled=(not selected or single_class)):
        with st.spinner("Melatih model..."):
            models = get_models(selected, knn_k, max_depth, balanced, seed)
            res = train_and_evaluate(models, X_train, X_test, y_train, y_test,
                                     scaler_name, cv_folds, seed)
        st.session_state["results"] = dict(
            res=res, config=config, y_test=y_test, X_test=X_test, y_train=y_train)
    state = st.session_state.get("results")
    if state:
        if state["config"] != config:
            st.warning("Konfigurasi sidebar berubah sejak training terakhir. Klik **Train models** "
                       "lagi agar hasil sesuai konfigurasi saat ini.")
        lb = ev.leaderboard(state["res"])
        fmt = {c: "{:.4f}" for c in ev.METRIC_COLS}

        def hl(col):
            key = "_cv_mean" if col.name == "CV Accuracy (mean ± std)" else col.name
            best = lb[key].max()
            return ["background-color: #c8e6c9; font-weight: bold" if lb.loc[i, key] == best else ""
                    for i in col.index]

        styled = (lb.drop(columns="_cv_mean").style
                  .apply(hl, subset=ev.METRIC_COLS + ["CV Accuracy (mean ± std)"])
                  .format(fmt))
        st.subheader("Quick Leaderboard")
        st.dataframe(styled, width='stretch')
        st.caption("Hijau = nilai terbaik per kolom. Metrik holdout dihitung pada test set; "
                   "CV Accuracy pada data latih.")
    else:
        st.write("Belum ada hasil. Klik **Train models**.")

# ------------------------------------------------------------------ 5. Evaluation
with tabs[4]:
    st.caption("Fase CRISP-DM: Evaluation.")
    state = st.session_state.get("results")
    if not state:
        st.info("Latih model terlebih dahulu di tab Modeling.")
    else:
        res, y_te, X_te = state["res"], state["y_test"], state["X_test"]
        if state["config"] != config:
            st.warning("Hasil di bawah berasal dari konfigurasi sebelumnya (sidebar telah berubah).")
        if state["config"]["scenario"] == SCENARIO_B:
            st.warning(f"Skenario B: skor tinggi sebagian besar disebabkan oleh `{LEAK_COL}` "
                       "(data leakage), karena progres kursus hampir identik dengan penyelesaian kursus.")
        lb = ev.leaderboard(res)

        st.subheader("Perbandingan Model")
        metric = st.selectbox("Metrik", ev.METRIC_COLS, index=3)
        fig, ax = new_fig()
        vals = lb[metric].sort_values()
        ax.barh(vals.index, vals.values, color="#4c72b0")
        for i, v in enumerate(vals.values):
            ax.text(v, i, f" {v:.3f}", va="center")
        ax.set(title=f"Perbandingan {metric}", xlabel=metric, ylabel="Model",
               xlim=(0, min(1.15, vals.max() * 1.15 + 0.05)))
        show(fig)

        model_name = st.selectbox("Pilih model", list(res.keys()))
        r = res[model_name]
        left, right = st.columns(2)
        with left:
            st.subheader("Confusion Matrix")
            tp, fp, fn, tn = ev.confusion_parts(y_te, r["y_pred"])
            fig, ax = new_fig()
            sns.heatmap([[tp, fn], [fp, tn]], annot=[[f"TP\n{tp}", f"FN\n{fn}"], [f"FP\n{fp}", f"TN\n{tn}"]],
                        fmt="", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["Pred: Not Completed", "Pred: Completed"],
                        yticklabels=["Actual: Not Completed", "Actual: Completed"])
            ax.set_title(f"Confusion matrix — {model_name}")
            show(fig)
            st.caption("Positif = Not Completed (0).")
        with right:
            st.subheader("ROC Curve")
            fig, ax = new_fig()
            for n, rr in res.items():
                fpr, tpr, _ = ev.roc_points(y_te, rr["proba0"])
                ax.plot(fpr, tpr, label=f"{n} (AUC = {rr['metrics']['ROC-AUC']:.3f})")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
            ax.set(title="ROC curve (positif = Not Completed)", xlabel="False Positive Rate",
                   ylabel="True Positive Rate")
            ax.legend(loc="lower right", fontsize=8)
            show(fig)

        st.subheader("Classification Report")
        st.dataframe(ev.report_frame(y_te, r["y_pred"]).style.format("{:.3f}"),
                     width='stretch')

        st.subheader("Feature Importance")
        imp = ev.feature_importance(r["pipeline"], X_te, y_te, seed).head(15).iloc[::-1]
        fig, ax = new_fig()
        ax.barh(imp["feature"], imp["importance"], color="#55a868")
        method = imp["method"].iloc[0]
        ax.set(title=f"Feature importance ({method}) — {model_name}", xlabel="Importance", ylabel="Fitur")
        show(fig)

        st.subheader("Interpretasi")
        best = lb["Macro F1"].idxmax()
        st.write(f"Model terbaik menurut **Macro F1**: **{best}** ({lb.loc[best, 'Macro F1']:.3f}).")
        minority = int(state["y_train"].value_counts().idxmin())
        for n, rr in res.items():
            from sklearn.metrics import recall_score
            rec = recall_score(y_te, rr["y_pred"], pos_label=minority, zero_division=0)
            acc = rr["metrics"]["Accuracy"]
            if rec < 0.5 and acc > rec:
                st.warning(f"Accuracy paradox pada **{n}**: accuracy {acc:.3f} tampak baik, tetapi "
                           f"recall kelas minoritas ({CLASS_NAMES[minority]}) hanya {rec:.3f}.")

# ------------------------------------------------------------------ 6. Clustering
with tabs[5]:
    st.caption("Fase CRISP-DM: Modeling (deskriptif, k-means tanpa target).")
    cols = clu.ENGAGEMENT_COLS
    Xs = clu.scale_features(df, cols)
    st.write(f"Fitur: {', '.join(cols)} (selalu di-scale dengan StandardScaler; target tidak dipakai).")
    et = clu.elbow_table(Xs, seed)
    a, b = st.columns(2)
    with a:
        fig, ax = new_fig()
        ax.plot(et["k"], et["SSE"], "o-")
        ax.set(title="Elbow plot", xlabel="k", ylabel="SSE (inertia)")
        show(fig)
    with b:
        fig, ax = new_fig()
        ax.plot(et["k"], et["Silhouette"], "o-", color="#c44e52")
        ax.set(title="Silhouette score per k", xlabel="k", ylabel="Silhouette")
        show(fig)
    k = st.slider("Jumlah cluster (k)", 2, 10, 3)
    labels, pcs, pca = clu.fit_kmeans(Xs, k, seed)
    prof = clu.profile(df, labels, cols)
    seg = clu.name_segments(prof, cols)
    st.markdown("**Profil cluster**")
    show_prof = prof.rename(index=seg).sort_index()
    st.dataframe(show_prof.style.format({**{c: "{:.2f}" for c in cols},
                                         "Completion rate (%)": "{:.1f}"}),
                 width='stretch')
    fig, ax = new_fig((7, 5))
    seg_labels = pd.Series(labels).map(seg)
    sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=seg_labels, s=12, alpha=0.6, ax=ax)
    ev_ratio = pca.explained_variance_ratio_
    ax.set(title="Cluster pada ruang PCA 2D", xlabel=f"PC1 ({ev_ratio[0] * 100:.1f}%)",
           ylabel=f"PC2 ({ev_ratio[1] * 100:.1f}%)")
    ax.legend(title="Segmen", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    show(fig)
    st.caption("Nama segmen ditentukan otomatis dari rata-rata z-score engagement tiap cluster.")

