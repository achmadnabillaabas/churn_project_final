# src/tune_model.py
import os, glob, optuna
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

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

def get_X_y():
    if DATA_PATH is None:
        raise FileNotFoundError("No CSV dataset found in data/ folder.")
    df = load_data(DATA_PATH)
    df = preprocess(df)
    df = add_features(df)
    if 'Exited' not in df.columns:
        cols_clean = [c.strip() for c in df.columns.astype(str)]
        candidates = [c for c in cols_clean if c.lower() in ('exited','churn','target','y') or 'exit' in c.lower()]
        if not candidates:
            raise ValueError("Target column not found")
        original_target_col = df.columns[cols_clean.index(candidates[0])]
        df.rename(columns={original_target_col:'Exited'}, inplace=True)
    X = df.drop("Exited", axis=1)
    y = df["Exited"].astype(int)
    X_num = X.select_dtypes(include=['number']).copy()
    cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    if cat_cols:
        X_dummies = pd.get_dummies(X[cat_cols].astype(str), drop_first=True)
        X_num = pd.concat([X_num, X_dummies], axis=1)
    return X_num, y

def objective(trial):
    X, y = get_X_y()
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
    }
    model = RandomForestClassifier(**params, random_state=42, class_weight='balanced')
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    score = cross_val_score(model, X, y, cv=skf, scoring="f1").mean()
    return score

def run_study(n_trials=20):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    print("Best trial:", study.best_trial.params)
    return study

if __name__ == "__main__":
    run_study(10)
