# src/build_index.py
from __future__ import annotations

import os
import pandas as pd
from .etl import load_dataset
from .vectorizer import build_tfidf, save_tfidf

CSV_PATH = "data/itinerary_dataset.csv"
PARQUET_PATH = "data/itinerary_dataset.clean.parquet"


def load_clean_frame() -> pd.DataFrame:
    # Ưu tiên parquet sạch do ETL đã chuẩn hoá
    if os.path.exists(PARQUET_PATH):
        df = pd.read_parquet(PARQUET_PATH)
    else:
        df = load_dataset(CSV_PATH)

    if "text" not in df.columns:
        raise ValueError("Thiếu cột 'text' sau ETL.")

    # 🔧 Patch 1: đảm bảo index 0..N-1 để đồng bộ với ma trận TF-IDF
    df = df.reset_index(drop=True)
    return df


def main():
    df = load_clean_frame()
    texts = df["text"].astype(str).tolist()
    vec, X = build_tfidf(texts)
    save_tfidf(vec, X)
    print(f"✅ Built TF-IDF: {X.shape[0]} items, {X.shape[1]} terms")
    print("💾 Saved: models/tfidf_vectorizer.joblib & models/tfidf_matrix.npz")


if __name__ == "__main__":
    main()
