# src/train_model.py
import joblib
import pandas as pd      # <-- WAJIB ADA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os, glob

from src.preprocess import load_data, preprocess
from src.feature_engineering import add_features


def find_dataset_in_datafolder():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    candidates = []
    for pattern in ["*.csv", "*.CSV"]:
        candidates.extend(glob.glob(os.path.join(data_dir, pattern)))
    preferred = ["Churn_Modelling.csv", "churn_dataset.csv", "churn_first20.csv"]
    for p in preferred:
        ppath = os.path.join(data_dir, p)
        if ppath in candidates:
            return ppath
    return candidates[0] if candidates else None

DATA_PATH = find_dataset_in_datafolder()
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "churn_model.pkl")

def train():
    if DATA_PATH is None:
        raise FileNotFoundError("No CSV dataset found in data/ folder. Place your CSV there.")

    print("Loading dataset:", DATA_PATH)
    df = load_data(DATA_PATH)
    df = preprocess(df)
    df = add_features(df)

    # auto-detect target column
    cols_clean = [c.strip() for c in df.columns.astype(str)]
    target_candidates = [c for c in cols_clean if c.lower() in ("exited","target","churn","y") or "exit" in c.lower()]
    if len(target_candidates) == 0:
        print("Columns found:", cols_clean)
        raise ValueError("Target column 'Exited' not found. Please ensure CSV has a target column.")
    target_col_clean = target_candidates[0]
    original_target_col = df.columns[cols_clean.index(target_col_clean)]
    df.rename(columns={original_target_col: "Exited"}, inplace=True)

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    print("Target distribution:", y.value_counts().to_dict())

    if len(y.value_counts()) == 1:
        raise ValueError("Target has only one class. Cannot train classifier.")

    stratify_arg = y if y.value_counts().min() >= 2 else None
    if stratify_arg is None:
        print("Warning: Some classes have <2 samples. Proceeding without stratify.")

    # minimal encoding
    X_numeric = X.select_dtypes(include=['number']).copy()
    cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    if cat_cols:
        X_dummies = pd.get_dummies(X[cat_cols].astype(str), drop_first=True)
        X_numeric = pd.concat([X_numeric, X_dummies], axis=1)

    if X_numeric.shape[1] == 0:
        raise ValueError("No numeric or encoded features available after preprocessing. Check your CSV.")

    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, stratify=stratify_arg, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Model saved to:", MODEL_PATH)

if __name__ == "__main__":
    train()
