# src/evaluate.py
import joblib
import os
import pandas as pd   # <-- WAJIB DITAMBAHKAN
from sklearn.metrics import classification_report, confusion_matrix
from src.preprocess import load_data, preprocess
from src.feature_engineering import add_features

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "churn_model.pkl")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def evaluate():
    import glob
    candidates = glob.glob(os.path.join(DATA_PATH, "*.csv"))
    if not candidates:
        raise FileNotFoundError("No CSV found in data/")

    df = load_data(candidates[0])
    df = preprocess(df)
    df = add_features(df)

    if "Exited" not in df.columns:
        cols_clean = [c.strip() for c in df.columns.astype(str)]
        candidates_col = [c for c in cols_clean if c.lower() in ("exited","churn","target","y") or "exit" in c.lower()]
        if not candidates_col:
            raise ValueError("Target column not found")
        original_target_col = df.columns[cols_clean.index(candidates_col[0])]
        df.rename(columns={original_target_col: "Exited"}, inplace=True)

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # Apply encoding
    X_num = X.select_dtypes(include=["number"]).copy()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    if cat_cols:
        X_num = X_num.join(pd.get_dummies(X[cat_cols].astype(str), drop_first=True))

    model = joblib.load(MODEL_PATH)

    preds = model.predict(X_num)

    print(classification_report(y, preds))
    print(confusion_matrix(y, preds))

if __name__ == "__main__":
    evaluate()
