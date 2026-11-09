# Online Course Engagement — Data Mining Testbed

Web app Streamlit: pipeline Overview → EDA → Preprocessing → Modeling → Evaluation → Clustering
pada dataset [Predict Online Course Engagement](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-course-engagement-dataset).
Target: `CourseCompletion` (apakah siswa menyelesaikan kursus). Semua parameter diatur dari sidebar.

## Instalasi (lokal)
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Menjalankan
```bash
streamlit run app.py
```
Buka http://localhost:8501. Dataset default ada di `data/online_course_engagement_data.csv`;
CSV lain bisa di-upload dari sidebar (wajib memiliki 9 kolom yang sama).

## Deploy ke Streamlit Community Cloud
1. Push folder proyek (termasuk `data/` dan `requirements.txt`) ke repository GitHub.
2. Buka https://share.streamlit.io, login dengan GitHub, klik **Create app**.
3. Pilih repository dan branch, isi **Main file path** = `app.py`, lalu **Deploy**.
4. Pilih versi Python di *Advanced settings* yang kompatibel dengan versi paket di `requirements.txt`
   (dikembangkan dengan Python 3.14; jika gagal, turunkan pin versi paket sesuai Python yang dipilih).

## Struktur
`app.py` (UI), `src/data.py`, `preprocessing.py`, `models.py`, `evaluation.py`, `clustering.py`.

## Catatan
- `UserID` di-drop; `CompletionRate` kandidat leakage (skenario A drop, B keep).
- Semua preprocessing berada di `sklearn.Pipeline` dan hanya di-fit pada data latih.
- Kelas positif untuk metrik early warning adalah 0 (Not Completed).
