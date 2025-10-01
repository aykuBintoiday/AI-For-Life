# src/etl_qna.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .textnorm import strip_accents
except Exception:
    import unicodedata
    def strip_accents(s: str) -> str:
        s = str(s or "")
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return s

def parse_hhmm(s: str) -> float:
    if s is None:
        return np.nan
    t = str(s).strip()
    if not t or t in {"-", "—", "–", "na", "n/a", "null"}:
        return np.nan
    try:
        parts = t.replace(".", ":").split(":")
        h = int(parts[0]); m = int(parts[1]) if len(parts) > 1 else 0
        v = 60*h + m
        if v < 0 or v >= 24*60:
            return np.nan
        return float(v)
    except Exception:
        return np.nan

SCHEMA = ["id","name","city","address","open_time","close_time","best_time","desc"]

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in SCHEMA:
        if c not in df.columns:
            df[c] = np.nan
    return df[SCHEMA].copy()

def main(in_csv: str | None = None, out_parquet: str | None = None):
    ROOT = Path(__file__).resolve().parents[1]
    in_csv = in_csv or str(ROOT / "data" / "qna_places.csv")
    out_parquet = out_parquet or str(ROOT / "data" / "qna_places.clean.parquet")

    p = Path(in_csv)
    if not p.exists():
        # If no CSV, create empty parquet (so downstream still works)
        df = pd.DataFrame(columns=SCHEMA)
    else:
        df = pd.read_csv(p, encoding="utf-8-sig", on_bad_lines="skip")
        df = _ensure_cols(df)

    # normalize
    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype("int64")
    df["name"] = df["name"].astype(str).fillna("").str.strip()
    df["city"] = df["city"].astype(str).fillna("").str.strip()
    df["address"] = df["address"].astype(str).fillna("").str.strip()
    df["open_time"] = df["open_time"].astype(str).fillna("").str.strip()
    df["close_time"] = df["close_time"].astype(str).fillna("").str.strip()
    df["best_time"] = df["best_time"].astype(str).fillna("").str.strip()
    df["desc"] = df["desc"].astype(str).fillna("").str.strip()

    df["open_min"] = df["open_time"].apply(parse_hhmm)
    df["close_min"] = df["close_time"].apply(parse_hhmm)

    df["norm_name"] = df["name"].map(lambda s: strip_accents(s).lower().strip())
    df["text"] = (df["name"].fillna("") + " | " +
                  df["address"].fillna("") + " | " +
                  df["best_time"].fillna("") + " | " +
                  df["desc"].fillna("")).str.strip()

    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"ETL QnA → {out_parquet} rows={len(df)}")

if __name__ == "__main__":
    # CLI: python -m src.etl_qna [in_csv] [out_parquet]
    in_csv = sys.argv[1] if len(sys.argv) >= 2 else None
    out_pq = sys.argv[2] if len(sys.argv) >= 3 else None
    main(in_csv, out_pq)
