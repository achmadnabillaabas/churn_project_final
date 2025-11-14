# src/preprocess.py
import pandas as pd
import csv

def _try_read_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs)

def load_data(path: str) -> pd.DataFrame:
    # Try default read
    try:
        df = _try_read_csv(path)
    except Exception:
        df = pd.read_csv(path, engine='python')

    # If only 1 column, try semicolon
    if df.shape[1] == 1:
        try:
            df2 = _try_read_csv(path, sep=';')
            if df2.shape[1] > 1:
                df = df2
        except Exception:
            pass

    # Try csv.Sniffer if still 1 column
    if df.shape[1] == 1:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                sample = f.read(2048)
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)
                delim = dialect.delimiter
            df3 = _try_read_csv(path, sep=delim)
            if df3.shape[1] > 1:
                df = df3
        except Exception:
            # fallback manual split on ';'
            try:
                text = open(path, 'r', encoding='utf-8', errors='ignore').read()
                lines = [r for r in text.splitlines() if r.strip() != ""]
                rows = [line.split(';') for line in lines]
                header = [h.strip() for h in rows[0]]
                data = rows[1:]
                df = pd.DataFrame(data, columns=header)
            except Exception:
                pass

    # normalize column names
    df.rename(columns={c: c.strip() for c in df.columns.astype(str)}, inplace=True)

    # drop fully empty rows
    df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
    df.dropna(how='all', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # attempt numeric conversion only for columns that look numeric
    for col in df.columns:
        s = df[col].astype(str).str.replace(',', '').str.strip()
        coerced = pd.to_numeric(s, errors='coerce')
        non_na = coerced.notna().sum()
        if non_na >= max(2, int(0.8 * len(df))):
            df[col] = coerced
        else:
            df[col] = df[col].astype(str).str.strip()

    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["RowNumber", "CustomerId", "Surname"]:
        if c in df.columns:
            try:
                df.drop(columns=[c], inplace=True)
            except Exception:
                pass
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    return df
