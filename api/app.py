# api/app.py
from fastapi import FastAPI
import joblib, os, pandas as pd
app = FastAPI()
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "churn_model.pkl")
model = joblib.load(MODEL_PATH)
@app.get("/")
def root():
    return {"status":"ok"}
@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    X_num = df.select_dtypes(include=['number']).copy()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    if cat_cols:
        X_num = X_num.join(pd.get_dummies(df[cat_cols].astype(str), drop_first=True))
    pred = int(model.predict(X_num)[0])
    proba = float(model.predict_proba(X_num)[0,1]) if hasattr(model,'predict_proba') else None
    return {"prediction": pred, "probability": proba}
