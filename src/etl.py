# src/etl.py
from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# if your project already has these helpers, import; otherwise keep the fallbacks
try:
    from .textnorm import strip_accents
except Exception:
    import unicodedata
    def strip_accents(s: str) -> str:
        s = str(s or "")
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return s

def parse_closed_days(v) -> set[int]:
    """
    Accepts: '0,6' or 'Sun;Sat' or [0,6] or '' -> set()
    0=Mon ... 6=Sun (keep consistent with your codebase).
    """
    if isinstance(v, set):
        return v
    if isinstance(v, (list, tuple)):
        out = set()
        for x in v:
            try:
                out.add(int(x))
            except Exception:
                pass
        return out
    if not isinstance(v, str):
        return set()
    s = v.strip()
    if not s:
        return set()
    names = {
        "mon": 0, "monday": 0, "0": 0,
        "tue": 1, "tuesday": 1, "1": 1,
        "wed": 2, "wednesday": 2, "2": 2,
        "thu": 3, "thursday": 3, "3": 3,
        "fri": 4, "friday": 4, "4": 4,
        "sat": 5, "saturday": 5, "5": 5,
        "sun": 6, "sunday": 6, "6": 6,
    }
    out = set()
    for tok in s.replace(";", ",").split(","):
        k = strip_accents(tok).lower().strip()
        if k in names:
            out.add(names[k])
        else:
            try:
                out.add(int(k))
            except Exception:
                pass
    return out

def parse_hhmm(s: str) -> float:
    """
    '15:00' -> 900; '23:59' -> 1439; '', '-', 'N/A' -> np.nan
    """
    if s is None:
        return np.nan
    t = str(s).strip()
    if not t or t in {"-", "—", "–", "na", "n/a", "null"}:
        return np.nan
    try:
        parts = t.replace(".", ":").split(":")
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        v = 60 * h + m
        if v < 0 or v >= 24 * 60:
            return np.nan
        return float(v)
    except Exception:
        return np.nan

def _safe_float(x, default=0.0):
    try:
        if x in ("", None, "—", "–", "-"):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def _safe_int(x, default=0):
    try:
        if x in ("", None, "—", "–", "-"):
            return int(default)
        return int(float(x))
    except Exception:
        return int(default)

SCHEMA = [
    "id", "name", "city", "category",
    "avg_cost", "duration_min", "price_level",
    "open_time", "close_time", "closed_days",
    "rating", "review_count", "best_time",
    "desc", "tags", "image_url", "lat", "lon", "address",
]

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in SCHEMA:
        if c not in df.columns:
            df[c] = np.nan
    return df[SCHEMA].copy()

def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        # fallback: use the last clean parquet if available, else empty frame
        pq = p.parent.parent / "data" / "itinerary_dataset.clean.parquet"
        if pq.exists():
            return pd.read_parquet(pq)
        return pd.DataFrame(columns=SCHEMA)
    df = pd.read_csv(p, encoding="utf-8-sig", on_bad_lines="skip")
    return _ensure_cols(df)

def main(in_csv: str | None = None, out_parquet: str | None = None):
    ROOT = Path(__file__).resolve().parents[1]
    in_csv = in_csv or str(ROOT / "data" / "itinerary_dataset.csv")
    out_parquet = out_parquet or str(ROOT / "data" / "itinerary_dataset.clean.parquet")

    df = load_dataset(in_csv)

    # types & normalization
    df["id"] = df["id"].apply(_safe_int)
    df["name"] = df["name"].astype(str).fillna("").str.strip()
    df["city"] = df["city"].astype(str).fillna("").str.strip()
    df["category"] = df["category"].astype(str).fillna("").str.strip()

    df["avg_cost"] = df["avg_cost"].apply(_safe_float)
    df["duration_min"] = df["duration_min"].apply(_safe_float)
    df["price_level"] = df["price_level"].astype(str).fillna("")

    df["open_time"] = df["open_time"].astype(str).fillna("")
    df["close_time"] = df["close_time"].astype(str).fillna("")
    df["open_min"] = df["open_time"].apply(parse_hhmm)
    df["close_min"] = df["close_time"].apply(parse_hhmm)

    df["closed_days"] = df["closed_days"].apply(parse_closed_days)
    df["closed_set"] = df["closed_days"].apply(parse_closed_days)

    df["rating"] = df["rating"].apply(_safe_float)
    df["review_count"] = df["review_count"].apply(_safe_int)
    df["best_time"] = df["best_time"].astype(str).fillna("")
    df["desc"] = df["desc"].astype(str).fillna("")
    df["tags"] = df["tags"].astype(str).fillna("")
    df["image_url"] = df["image_url"].astype(str).fillna("")
    df["address"] = df["address"].astype(str).fillna("")

    df["lat"] = df["lat"].apply(_safe_float)
    df["lon"] = df["lon"].apply(_safe_float)

    # helpers for search
    df["norm_name"] = df["name"].map(lambda s: strip_accents(s).lower().strip())
    df["text"] = (df["name"].fillna("") + " | " +
                  df["category"].fillna("") + " | " +
                  df["tags"].fillna("") + " | " +
                  df["address"].fillna("") + " | " +
                  df["desc"].fillna("")).str.strip()

    # write
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"ETL itinerary → {out_parquet} rows={len(df)}")

if __name__ == "__main__":
    # CLI: python -m src.etl [in_csv] [out_parquet]
    in_csv = sys.argv[1] if len(sys.argv) >= 2 else None
    out_pq = sys.argv[2] if len(sys.argv) >= 3 else None
    main(in_csv, out_pq)
