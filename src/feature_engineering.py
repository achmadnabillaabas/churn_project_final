# src/feature_engineering.py
import numpy as np
import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Balance' in df.columns and 'EstimatedSalary' in df.columns:
        # avoid division by zero
        df['balance_ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    if 'Age' in df.columns and 'NumOfProducts' in df.columns:
        df['age_x_products'] = df['Age'] * df['NumOfProducts']
    if 'Tenure' in df.columns and 'IsActiveMember' in df.columns:
        df['tenure_x_active'] = df['Tenure'] * df['IsActiveMember']
    if 'Balance' in df.columns:
        df['log_balance'] = np.log1p(df['Balance'].fillna(0))
    return df
