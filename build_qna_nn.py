# build_qna_nn.py
from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
PQ   = ROOT / "data" / "qna_places.clean.parquet"
OUT_E = ROOT / "models" / "qna_embeds.npy"
OUT_M = ROOT / "models" / "qna_meta.json"

def main():
    if not PQ.exists():
        raise FileNotFoundError(f"Missing {PQ}. Run: .\\.venv\\Scripts\\python.exe -m src.etl_qna .\\data\\qna_places.csv")

    df = pd.read_parquet(PQ)
    texts = df["text"].astype(str).fillna("").tolist()
    rows = len(texts)
    print(f"[qna_nn] rows={rows}")

    # chọn model: ưu tiên theo ENV NN_MODEL, sau đó theo meta itinerary (nếu muốn đồng bộ),
    # cuối cùng default e5-base
    model_name = os.getenv("NN_MODEL", "").strip()
    if not model_name:
        try:
            meta_it = json.loads((ROOT / "models" / "nn_meta.json").read_text(encoding="utf-8"))
            model_name = meta_it.get("model", "")
        except Exception:
            model_name = ""
    if not model_name:
        model_name = "intfloat/multilingual-e5-base"

    print(f"[qna_nn] model={model_name}")
    nn = SentenceTransformer(model_name)

    embs = []
    BS = 512
    for i in tqdm(range(0, rows, BS), desc="Batches"):
        batch = texts[i:i+BS]
        e = nn.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        embs.append(e.astype("float32"))
    E = np.vstack(embs)
    OUT_E.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_E, E)
    OUT_M.write_text(json.dumps({"model": model_name, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[qna_nn] saved {OUT_E.name} ({E.shape[0]}x{E.shape[1]}) and {OUT_M.name}")

if __name__ == "__main__":
    main()
