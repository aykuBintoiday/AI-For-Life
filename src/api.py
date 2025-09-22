# src/api.py
from __future__ import annotations

import os
import logging
import traceback
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity  # <- thêm cho /_debug/preview

from .recommender import Recommender

# ==== logging (dev) ====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("travel-ai-api")


def tb_string(limit: int = 2000) -> str:
    """Chuỗi traceback rút gọn để trả về trong detail (dev only)."""
    s = traceback.format_exc()
    if len(s) > limit:
        s = s[:limit] + "...(truncated)"
    return s


app = FastAPI(title="Travel Planner (Cosine + ML + Heuristic)")

# CORS mở cho demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo recommender
try:
    rec = Recommender()
except Exception as e:
    logger.exception("Failed to init Recommender")
    raise


class ItineraryResp(BaseModel):
    days: int
    per_day_budget: int
    total_cost_estimate: int
    itinerary: list


@app.get("/ping")
def ping():
    return {"ok": True}


@app.get("/place")
def place(name: str = Query(..., description="Tên địa điểm (copy/paste)")):
    try:
        r = rec.get_place(name)
        if not r:
            raise HTTPException(status_code=404, detail="Không tìm thấy địa điểm")
        return r
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Place endpoint error")
        raise HTTPException(
            status_code=400,
            detail={
                "msg": f"Place lookup failed: {type(e).__name__}: {e}",
                "traceback": tb_string(),
                "hint": "Kiểm tra name (có dấu/không dấu). Ví dụ thử 'Ba Na Hills' hoặc 'Bà Nà Hills'.",
            },
        )


@app.get("/itinerary", response_model=ItineraryResp)
def itinerary(
    q: str = Query(..., description="Sở thích/mô tả chuyến đi (tự nhiên)"),
    days: int = Query(..., ge=1, le=10),
    budget_total: int = Query(..., ge=0),
    city: Optional[str] = Query(None),
):
    """
    Dev-friendly: Trước khi gọi recommender, kiểm tra các điều kiện hay lỗi thường gặp.
    """
    try:
        # --- Preflight checks: dữ liệu & TF-IDF có khớp không?
        df_len = len(rec.df) if getattr(rec, "df", None) is not None else None
        X_shape = tuple(getattr(rec, "X", None).shape) if getattr(rec, "X", None) is not None else None

        if df_len is None or X_shape is None:
            raise RuntimeError("Recommender chưa load đủ df hoặc TF-IDF (X).")

        if X_shape[0] != df_len:
            raise RuntimeError(
                f"TF-IDF matrix rows ({X_shape[0]}) != DataFrame rows ({df_len}). "
                f"Cần rebuild TF-IDF sau ETL."
            )

        # --- Gọi recommender
        result = rec.itinerary(q, days, budget_total, city)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Itinerary endpoint error")
        # Gợi ý khắc phục theo từng loại lỗi thường gặp
        hint = "Xem traceback, sau đó thử: python -m src.build_index (rebuild TF-IDF), rồi chạy lại server."
        if "TF-IDF" in str(e) or "matrix" in str(e) or "shape" in str(e) or "index" in str(e).lower():
            hint = (
                "Khả năng lệch index giữa DataFrame và TF-IDF. "
                "Hãy đảm bảo trong build_index có reset_index(drop=True), "
                "recommender cũng reset_index(drop=True), và trong cosine dùng cos_all[df.index]. "
                "Sau đó chạy: python -m src.build_index"
            )
        elif "city" in str(e).lower():
            hint = "Thử bỏ dấu ở city (vd: 'Da Nang') hoặc bỏ hẳn tham số city để test."
        elif "models" in str(e).lower() or "joblib" in str(e).lower():
            hint = "Thiếu models? Kiểm tra thư mục models/ có tfidf_vectorizer.joblib, tfidf_matrix.npz, ml_classifier.joblib chưa."

        raise HTTPException(
            status_code=400,
            detail={
                "msg": f"Build itinerary failed: {type(e).__name__}: {e}",
                "traceback": tb_string(),
                "hint": hint,
                "echo": {
                    "q": q,
                    "days": days,
                    "budget_total": budget_total,
                    "city": city,
                },
            },
        )


# =========================
#        DEBUG ENDPOINTS
# =========================

@app.get("/_debug/state")
def debug_state():
    """
    Trả về trạng thái hiện tại của dữ liệu & TF-IDF để chẩn đoán nhanh.
    Không nên bật endpoint này ở production.
    """
    try:
        df = getattr(rec, "df", None)
        X = getattr(rec, "X", None)
        vec = getattr(rec, "vec", None)

        info = {
            "df_rows": len(df) if df is not None else None,
            "df_cols": list(df.columns)[:20] if df is not None else None,
            "has_norm_name": bool("norm_name" in df.columns) if df is not None else None,
            "has_text": bool("text" in df.columns) if df is not None else None,
            "X_shape": tuple(X.shape) if X is not None else None,
            "vocabulary_size": len(getattr(vec, "vocabulary_", {})) if vec is not None else None,
            "index_match": (tuple(X.shape)[0] == len(df)) if (X is not None and df is not None) else None,
            "city_samples": df["city"].value_counts().to_dict() if df is not None and "city" in df.columns else None,
            "category_samples": df["category"].value_counts().to_dict() if df is not None and "category" in df.columns else None,
        }
        # Gợi ý tự động
        hints = []
        if info["df_rows"] is None or info["X_shape"] is None:
            hints.append("rec.df hoặc rec.X chưa sẵn sàng. Kiểm tra khởi tạo Recommender.")
        elif info["index_match"] is False:
            hints.append("TF-IDF rows != DF rows. Rebuild TF-IDF: python -m src.build_index")
        if not info["has_text"]:
            hints.append("Thiếu cột 'text' sau ETL. Chạy lại: python -m src.etl data/itinerary_dataset.csv")

        info["hints"] = hints
        return info
    except Exception as e:
        logger.exception("/_debug/state failed")
        raise HTTPException(
            status_code=500,
            detail={"msg": f"debug_state failed: {type(e).__name__}: {e}", "traceback": tb_string()},
        )


@app.get("/_debug/files")
def debug_files():
    """
    Kiểm tra sự tồn tại các file model và kích thước.
    """
    try:
        paths = [
            "models/tfidf_vectorizer.joblib",
            "models/tfidf_matrix.npz",
            "models/ml_classifier.joblib",
            "data/itinerary_dataset.csv",
            "data/itinerary_dataset.clean.parquet",
        ]
        files = []
        for p in paths:
            files.append({
                "path": p,
                "exists": os.path.exists(p),
                "size_bytes": os.path.getsize(p) if os.path.exists(p) else None
            })
        return {"files": files}
    except Exception as e:
        logger.exception("/_debug/files failed")
        raise HTTPException(
            status_code=500,
            detail={"msg": f"debug_files failed: {type(e).__name__}: {e}", "traceback": tb_string()},
        )


@app.get("/debug/status")
def debug_status():
    try:
        n, v = rec.X.shape
    except Exception:
        n, v = None, None
    return {
        "rows": len(rec.df),
        "tfidf": {"n_items": n, "n_terms": v},
        "cities": sorted(list(rec.df["city"].unique())),
        "cats": rec.df["category"].value_counts().to_dict(),
    }


@app.get("/_debug/preview")
def debug_preview(q: str = Query(..., description="Câu hỏi gốc của người dùng")):
    """
    Cho biết query sau khi normalize + top 10 item theo cosine để soi nhanh vì sao 'không hiểu chữ'.
    """
    try:
        q_norm = rec.normalize_for_debug(q)
        import numpy as np
        q_vec = rec.vec.transform([q_norm])
        cos = cosine_similarity(q_vec, rec.X).ravel()
        topk_idx = np.argsort(-cos)[:10].tolist()
        top = rec.df.iloc[topk_idx][["id", "name", "city", "category"]].copy()
        top["cos"] = [float(cos[i]) for i in topk_idx]
        return {"q_norm": q_norm, "top": top.to_dict(orient="records")}
    except Exception as e:
        logger.exception("/_debug/preview failed")
        raise HTTPException(
            status_code=500,
            detail={"msg": f"debug_preview failed: {type(e).__name__}: {e}", "traceback": tb_string()},
        )

# --- THÊM VÀO CUỐI FILE api.py ---
from fastapi import Body

@app.post("/_debug/preview")
def debug_preview(payload: dict = Body(...)):
    """
    Preview khớp TF-IDF cho truy vấn q (và city tuỳ chọn).
    Trả về q_norm, q_vec.nnz và top 10 item với các điểm số.
    """
    try:
        q = str(payload.get("q", "") or "")
        city = payload.get("city")
        # Lọc city như trong recommender
        df = rec.df.copy()
        if city:
            key = rec.__class__.__mro__[0].__dict__.get("_city_filter", None)
        # dùng method của rec nếu có
        try:
            df = rec._city_filter(df, city) if city else df
        except Exception:
            pass

        # chuẩn hoá + vector hoá
        q_norm = rec.normalize_for_debug(q) if hasattr(rec, "normalize_for_debug") else q
        q_vec = rec.vec.transform([q_norm])
        cos_all = cosine_similarity(q_vec, rec.X).ravel()
        df = df.assign(cos=cos_all[df.index])

        # backoff kw_boost như trong recommender
        try:
            kw_boost = rec._kw_boost(df, q_norm)
            df["kw_boost"] = kw_boost
        except Exception:
            df["kw_boost"] = 0.0

        def base_row(x):
            return 0.70 * x["cos"] + 0.20 * (x.get("rating", 0) / 5) + 0.10 * (x.get("popularity", 0)) + 0.05 * x.get("kw_boost", 0)

        df["base"] = df.apply(base_row, axis=1)
        out = df.sort_values("base", ascending=False).head(10)[
            ["id","name","city","category","cos","kw_boost","rating","popularity","tags"]
        ].to_dict(orient="records")

        return {
            "q": q,
            "q_norm": q_norm,
            "q_vec_nnz": int(q_vec.nnz),
            "top10": out
        }
    except Exception as e:
        logger.exception("/_debug/preview failed")
        return {"error": f"{type(e).__name__}: {e}"}
