# build_nn.py
from __future__ import annotations
import os, json, math
import numpy as np
import pandas as pd

# Ưu tiên dùng CPU cho ổn định demo
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import SentenceTransformer

CSV_PATH = "data/itinerary_dataset.csv"
PARQUET_PATH = "data/itinerary_dataset.clean.parquet"
EMB_PATH = "models/nn_embeds.npy"
META_PATH = "models/nn_meta.json"

# Cho phép đổi model qua biến môi trường nếu muốn (ví dụ BAAI/bge-m3)
MODEL_NAME = os.getenv("NN_MODEL", "intfloat/multilingual-e5-base")
BATCH_SIZE = int(os.getenv("NN_BATCH", "64"))

def load_clean_frame() -> pd.DataFrame:
    if os.path.exists(PARQUET_PATH):
        df = pd.read_parquet(PARQUET_PATH)
    else:
        # fallback: đọc CSV thô và xử lý tối thiểu
        from src.etl import load_dataset
        df = load_dataset(CSV_PATH)

    # đảm bảo có cột text (đã gộp name/city/category/tags/desc + synonyms trong ETL)
    if "text" not in df.columns:
        raise ValueError("Thiếu cột 'text'. Hãy chạy: python -m src.etl data/itinerary_dataset.csv")
    # đồng bộ index 0..N-1 khớp mọi ma trận
    return df.reset_index(drop=True)

def main():
    os.makedirs("models", exist_ok=True)
    print(f"🧠 Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")  # demo an toàn CPU

    df = load_clean_frame()
    texts = df["text"].astype(str).tolist()
    n = len(texts)
    print(f"📄 Items: {n}")

    # Encode theo batch, chuẩn hoá L2 để cosine = dot
    all_embeds = []
    for i in range(0, n, BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        emb = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")
        all_embeds.append(emb)

    E = np.vstack(all_embeds) if all_embeds else np.zeros((0, 768), dtype="float32")
    if E.shape[0] != n:
        raise RuntimeError(f"Embedding rows mismatch: {E.shape[0]} vs df {n}")

    np.save(EMB_PATH, E)
    meta = {
        "model": MODEL_NAME,
        "rows": int(n),
        "dim": int(E.shape[1]) if E.ndim == 2 else None,
        "text_col": "text",
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved embeddings: {EMB_PATH} shape={E.shape}")
    print(f"✅ Saved meta: {META_PATH}")

if __name__ == "__main__":
    main()
