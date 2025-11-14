# Churn Project
Abstrak :

Penelitian ini membahas pengembangan dan evaluasi model prediksi churn menggunakan algoritma Random Forest pada dataset Churn Modelling. Model yang dikembangkan dibandingkan dengan penelitian pada Jurnal KONTRUKSI (2024). Hasil penelitian menunjukkan peningkatan performa model dengan akurasi mencapai 97% dan F1-score sebesar 93%, lebih tinggi dibandingkan penelitian sebelumnya yang memperoleh akurasi 91% dan F1-score 91%. Peningkatan performa disebabkan oleh optimasi preprocessing, feature engineering, dan penyesuaian hyperparameter. Artikel ini juga menyajikan grafik perbandingan serta output evaluasi lengkap.


Folder structure:
- data/: place your CSV dataset(s) here (e.g. Churn_Modelling.csv)
- src/: source code (preprocess, feature engineering, train, tune, evaluate)
- models/: trained models saved here
- api/: FastAPI app for prediction

Usage:
1. Create virtualenv and install requirements:
   python -m venv .venv
   .venv\Scripts\activate     # Windows PowerShell
   python -m pip install -r requirements.txt

2. Train model:
   python -m src.train_model

3. Tune model (Optuna):
   python -m src.tune_model

4. Run API:
   uvicorn api.app:app --reload --port 8000
