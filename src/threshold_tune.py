# src/threshold_tune.py
import joblib
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score, average_precision_score

model = joblib.load("models/churn_model.pkl")

# load data same as evaluate pipeline
from src.preprocess import load_data, preprocess
from src.feature_engineering import add_features
import glob, os, pandas as pd

data_path = glob.glob(os.path.join("data","*.csv"))[0]
df = load_data(data_path)
df = preprocess(df)
df = add_features(df)

# ensure target column named 'Exited'
cols_clean = [c.strip() for c in df.columns.astype(str)]
if 'Exited' not in df.columns:
    candidates = [c for c in cols_clean if c.lower() in ('exited','churn','target','y') or 'exit' in c.lower()]
    df.rename(columns={df.columns[cols_clean.index(candidates[0])]: 'Exited'}, inplace=True)

X = df.drop("Exited", axis=1)
y = df["Exited"]

# same minimal transform as evaluate.py
X_num = X.select_dtypes(include=['number']).copy()
cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()
if cat_cols:
    X_num = X_num.join(pd.get_dummies(X[cat_cols].astype(str), drop_first=True))

probs = model.predict_proba(X_num)[:,1]

best_f1 = 0
best_thr_f1 = 0
best_thr_recall = 0
best_recall = 0

for thr in np.linspace(0.05,0.95,91):
    preds = (probs >= thr).astype(int)
    f1 = f1_score(y, preds)
    rec = recall_score(y, preds)
    if f1 > best_f1:
        best_f1 = f1; best_thr_f1 = thr
    if rec > best_recall:
        best_recall = rec; best_thr_recall = thr

print("Best F1:", best_f1, "at thr", best_thr_f1)
print("Best Recall:", best_recall, "at thr", best_thr_recall)
# show confusion at chosen threshold (e.g. maximize recall threshold)
preds = (probs >= best_thr_recall).astype(int)
print("Confusion (thr recall):")
print(confusion_matrix(y, preds))
