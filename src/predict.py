# src/predict.py
import joblib, os, pandas as pd
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "churn_model.pkl")
model = joblib.load(MODEL_PATH)
def predict_single(data_dict):
    df = pd.DataFrame([data_dict])
    X_num = df.select_dtypes(include=['number']).copy()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    if cat_cols:
        X_num = X_num.join(pd.get_dummies(df[cat_cols].astype(str), drop_first=True))
    return model.predict(X_num)[0], (model.predict_proba(X_num)[0,1] if hasattr(model,'predict_proba') else None)
