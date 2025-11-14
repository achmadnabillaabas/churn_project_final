# Churn Project (auto-generated)
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
