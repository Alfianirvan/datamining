# CLAUDE.md — Online Course Engagement Data Mining Testbed

## Tujuan proyek

Bangun web app **Streamlit** berupa *data mining testbed*: pipeline end-to-end
(Overview → EDA → Preprocessing → Modeling → Evaluation → Clustering) pada dataset
**Predict Online Course Engagement** (Kaggle), yang semua parameternya bisa diatur dari sidebar.

Pertanyaan bisnis: **memprediksi apakah siswa menyelesaikan kursus online
(`CourseCompletion`) berdasarkan tingkat keterlibatan belajarnya**, supaya platform
bisa mendeteksi siswa berisiko tidak menyelesaikan kursus sejak dini.

Ini tugas kuliah Data Mining (Universitas Negeri Malang). Referensinya adalah contoh dosen
"Student Performance Data Mining Testbed" (UCI Student Performance). Struktur tab dan sidebar
dibuat mirip contoh itu, tetapi memakai dataset ini. Materi kuliah: CRISP-DM, data understanding,
preprocessing, klasifikasi (Naive Bayes, kNN, Decision Tree), evaluasi classifier, clustering (k-means).

## Dataset

- File: `data/online_course_engagement_data.csv` (sudah tersedia; jangan download dari internet)
- Sumber: https://www.kaggle.com/datasets/rabieelkharoua/predict-online-course-engagement-dataset
- 9.000 baris × 9 kolom, **tanpa missing value**

| Kolom | Tipe | Keterangan |
|---|---|---|
| `UserID` | int | ID siswa. **Bukan fitur, harus di-drop.** 877 ID muncul lebih dari sekali (tidak ada baris yang identik seluruhnya) |
| `CourseCategory` | kategorikal | Arts, Business, Health, Programming, Science |
| `TimeSpentOnCourse` | float | Jam belajar, 1–100 |
| `NumberOfVideosWatched` | int | 0–20 |
| `NumberOfQuizzesTaken` | int | 0–10 |
| `QuizScores` | float | 50–100 |
| `CompletionRate` | float | Persentase progres kursus, 0–100. **Kandidat data leakage** |
| `DeviceType` | int biner | 0 / 1 (anggap 0 = Desktop, 1 = Mobile; tulis asumsi ini di UI) |
| `CourseCompletion` | int biner | **TARGET**: 1 = selesai, 0 = tidak |

Fakta yang sudah dicek (pakai sebagai sanity check, jangan di-hardcode di UI; UI harus menghitung sendiri):
- Distribusi target: 0 = 5.432 (60,4%), 1 = 3.568 (39,6%), cukup tidak seimbang
- Korelasi dengan target: QuizScores 0,30; NumberOfQuizzesTaken 0,29; CompletionRate 0,33;
  DeviceType ≈ 0; tingkat penyelesaian per CourseCategory hampir sama (38–41%)
- 5-fold CV accuracy (baseline cepat):
  - Dengan CompletionRate: NB 0,82 · kNN 0,85 · DT(max_depth=5) 0,96
  - Tanpa CompletionRate: NB 0,77 · kNN 0,76 · DT 0,80

## Stack & struktur

- Python 3.10+, `streamlit`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`
- Buat `requirements.txt` dengan versi yang di-pin

```
project/
├── app.py                 # entry point Streamlit (sidebar + tab)
├── src/
│   ├── data.py            # load_data(), cached dengan st.cache_data
│   ├── preprocessing.py   # build_features(), build_preprocessor()
│   ├── models.py          # get_models(), train_and_evaluate()
│   ├── evaluation.py      # metrik, confusion matrix, ROC
│   └── clustering.py      # k-means, elbow, silhouette
├── data/online_course_engagement_data.csv
├── requirements.txt
└── README.md              # cara install, menjalankan, dan deploy
```

## Sidebar — "Testbed Configuration"

1. **Upload CSV (opsional)**: kalau ada upload, pakai file itu; kalau tidak, pakai file default.
   Validasi bahwa kolom wajib ada, dan tampilkan error yang jelas kalau tidak.
2. **Leakage scenario** (radio):
   - A: *Early warning*, drop `CompletionRate` (default)
   - B: *Progress tracking*, keep `CompletionRate`
3. **Feature scaler** (selectbox): StandardScaler (default), MinMaxScaler, None
4. **Test size (holdout)** (slider): 0,10–0,40, default 0,20
5. **Cross-validation folds (K)** (slider): 3–10, default 5
6. **Models** (multiselect): Naive Bayes, k-Nearest Neighbors, Decision Tree (ketiganya default
   terpilih), plus opsional Logistic Regression dan Random Forest
7. **Hyperparameter** (expander): k untuk kNN (1–25, default 5), max_depth Decision Tree
   (1–20 atau None, default 5)
8. **Handle class imbalance** (checkbox): kalau aktif, pakai `class_weight="balanced"` untuk model
   yang mendukung, dan stratified split/CV selalu dipakai
9. **Random seed** (number_input, default 42)

## Tab

Gunakan `st.tabs`. Label UI dalam **Bahasa Indonesia**; istilah teknis (accuracy, recall, dll.)
tetap Inggris. Setiap tab diawali satu kalimat `st.caption` yang menyebut fase CRISP-DM-nya.

### 1. Overview (Business & Data Understanding)
- Judul + deskripsi singkat tujuan bisnis
- Metric cards: Rows, Columns, Missing values, Completion rate (% target = 1), Duplicate UserID
- Tabel contoh data (`st.dataframe`, 10 baris) + tabel kamus data (kolom, tipe, keterangan)

### 2. EDA (Data Understanding)
- Statistik deskriptif (`describe`) + jenis tiap atribut (nominal/biner/rasio)
- Distribusi target (bar chart)
- Variabel numerik vs target: pilih variabel → histogram per kelas + boxplot,
  dan uji Mann-Whitney U atau t-test dengan p-value
- **Variabel kategorikal vs target (chi-square test)**: pilih `CourseCategory` / `DeviceType` →
  tabel kontingensi, stacked bar chart, chi-square statistic, p-value, dan kesimpulan
  "Signifikan / Tidak signifikan pada α = 0,05" (meniru contoh dosen)
- Heatmap korelasi antar variabel numerik

### 3. Preprocessing (Data Preparation)
- Tampilkan skenario, scaler, test size, dan CV folds yang aktif
- Langkah yang dijelaskan dan diterapkan: drop `UserID` (beri alasan), drop/keep `CompletionRate`
  sesuai skenario, one-hot encoding `CourseCategory`, scaling fitur numerik
- Semua transformasi harus di dalam `sklearn.Pipeline` + `ColumnTransformer`, di-fit **hanya
  pada data latih**, supaya tidak terjadi leakage
- Tampilkan "Shape after encoding: N rows × M features" + preview matriks fitur hasil transformasi
- Tampilkan ukuran train/test dan distribusi kelas di masing-masing (stratified)

### 4. Modeling
- Teks "Training on N selected model(s): …" + tombol **Train models**
- Simpan hasil di `st.session_state` supaya tab Evaluation bisa membacanya tanpa melatih ulang
- **Quick leaderboard** (tabel): Accuracy, Precision (Not Completed), Recall (Not Completed),
  Macro F1, ROC-AUC, CV Accuracy (mean ± std). Highlight nilai terbaik per kolom.
  Kelas yang dianggap "positif" untuk early warning = **0 (Not Completed)**; jelaskan di UI.

### 5. Evaluation
- Model comparison: dropdown metrik → horizontal bar chart semua model
- Confusion matrix per model (pilih model), dengan label TP/FP/FN/TN
- ROC curve semua model dalam satu grafik + AUC di legend
- Classification report per model
- Feature importance (Decision Tree / Random Forest) atau permutation importance
- Kotak "Interpretasi": model terbaik menurut macro F1, dan peringatan accuracy paradox kalau
  recall kelas minoritas rendah
- Kalau skenario B dipilih, tampilkan catatan bahwa skor tinggi disebabkan oleh `CompletionRate`
  (leakage)

### 6. Clustering (Descriptive, k-means)
- Fitur: variabel engagement numerik (tanpa target), selalu di-scale
- Elbow plot (SSE/inertia untuk k = 2–10) + silhouette score per k
- Slider k (default 3) → k-means (`n_init=10`, random_state = seed)
- Profil cluster: rata-rata tiap fitur per cluster + **completion rate per cluster**
- Scatter 2D hasil PCA, diwarnai per cluster
- Beri nama segmen otomatis berdasarkan rata-rata engagement (mis. "Aktif", "Sedang", "Pasif")

## Aturan kode

- Gunakan `@st.cache_data` untuk load data dan `@st.cache_resource` untuk hal yang berat
- Semua angka di UI dihitung dari data, tidak ada yang di-hardcode
- Selalu pakai `stratify=y` di `train_test_split` dan `StratifiedKFold` untuk CV
- Fungsi diberi docstring singkat; nama variabel dalam bahasa Inggris
- Grafik: matplotlib/seaborn dengan `st.pyplot(fig)`, ukuran konsisten, judul dan label sumbu jelas
- Tangani edge case: tidak ada model yang dipilih, file upload salah format, kelas hanya satu

## Definition of done

- [ ] `pip install -r requirements.txt && streamlit run app.py` jalan tanpa error/warning besar
- [ ] Keenam tab tampil dan semua kontrol sidebar benar-benar memengaruhi hasil
- [ ] Skenario A vs B menghasilkan akurasi yang jelas berbeda (sesuai baseline di atas, ± beberapa persen)
- [ ] Uji chi-square `CourseCategory` vs target menghasilkan "Tidak signifikan"
- [ ] Tidak ada leakage preprocessing (scaler/encoder di-fit di train saja, di dalam Pipeline)
- [ ] README menjelaskan cara menjalankan lokal dan deploy ke Streamlit Community Cloud
- [ ] Jalankan app dan cek setiap tab sendiri sebelum menyatakan selesai

## Cara kerja untuk Claude Code

1. Baca dataset dulu dan konfirmasi kolom/angka di atas.
2. Bangun bertahap: `src/` → tab Overview → EDA → Preprocessing → Modeling → Evaluation → Clustering.
   Jalankan app setelah setiap tab selesai.
3. Jangan menambah dependensi di luar stack tanpa alasan.
