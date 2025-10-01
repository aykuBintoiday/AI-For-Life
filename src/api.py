# src/api.py
from __future__ import annotations

import os
import logging
from typing import Optional

import numpy as np
from fastapi import FastAPI, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .recommender import Recommender
from .etl import strip_accents  # dùng ở /_debug/preview

# (tuỳ chọn) Q&A – khởi tạo mềm
try:
    from .qaguide import QAGuide  # noqa
except Exception:
    QAGuide = None  # type: ignore

log = logging.getLogger("travel-ai-api")
logging.basicConfig(level=logging.INFO)

# ---- Đọc token admin từ .env (đặt ADMIN_TOKEN=secret123) ----
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()

app = FastAPI(title="Travel AI API", version="1.0.0")

# CORS cho UI tĩnh
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # để test nhanh; production nên giới hạn domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Safe JSON ----------
def safe_json(obj):
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [safe_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [safe_json(v) for v in obj.tolist()]
    try:
        if obj is None:
            return None
        if isinstance(obj, (float, np.floating)):
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        return obj
    except Exception:
        return None

# ---------- Init models ----------
rec: Optional[Recommender] = None
qag = None

try:
    rec = Recommender()
    log.info("Recommender loaded")
except Exception:
    log.exception("Failed to init Recommender")
    rec = None

if QAGuide is not None and rec is not None:
    try:
        # truyền encoder của recommender cho QAGuide để dùng cùng model
        qag = QAGuide(encoder=rec.nn)
        log.info("QAGuide loaded")
    except Exception:
        log.exception("Failed to init QAGuide")
        qag = None


# ---------- Endpoints ----------
@app.get("/_health")
def health():
    return {"ok": True, "models": {"recommender": rec is not None, "qag": qag is not None}}


@app.get("/itinerary")
def itinerary(
    q: str = Query(..., description="Miêu tả mong muốn chuyến đi"),
    days: int = Query(2, ge=1, le=10),
    budget_total: int = Query(3_000_000, ge=0),
    city: Optional[str] = Query(None),
):
    if rec is None:
        return JSONResponse({"detail": "Recommender not available"}, status_code=503)
    try:
        data = rec.itinerary(q, days, budget_total, city)
        return JSONResponse(safe_json(data))
    except Exception:
        log.exception("Itinerary endpoint error")
        return JSONResponse({"detail": "Internal error while building itinerary"}, status_code=500)


@app.get("/_debug/preview")
def debug_preview(q: str, city: Optional[str] = None):
    """
    Trả nhanh top-10 theo nn_cos để kiểm tra liên quan (debug).
    """
    if rec is None:
        return JSONResponse({"detail": "Recommender not available"}, status_code=503)
    try:
        _, q_emb = rec._encode_query(q)
        nn = rec._bi_scores(q_emb)
        df = rec.df.copy()
        if city:
            key = df["city"].astype(str).str.lower().apply(strip_accents).str.contains(
                strip_accents(city).lower(), na=False
            )
            sub = df[key]
            if not sub.empty:
                df = sub
        df = df.assign(nn_cos=nn[df.index])
        top = df.sort_values("nn_cos", ascending=False).head(10)
        out = top[["id", "name", "city", "category", "nn_cos"]].to_dict(orient="records")
        return JSONResponse(safe_json({"q": q, "top10": out}))
    except Exception:
        log.exception("Preview error")
        return JSONResponse({"detail": "Preview failed"}, status_code=500)


@app.get("/ask")
def ask(
    q: str = Query(..., description="Câu hỏi về địa danh (ví dụ: 'Asia Park mở cửa mấy giờ?')"),
    city: Optional[str] = Query(None, description="Tùy chọn, để bias theo thành phố"),
):
    """
    Q&A hướng dẫn viên (nếu QAGuide init thành công).
    Ghi chú: QAGuide.answer chỉ nhận q; city (nếu có) sẽ được nối vào q cho mục đích bias nhẹ.
    """
    if qag is None:
        return JSONResponse({"detail": "QAGuide not available"}, status_code=503)
    try:
        q_in = f"{q} {city}" if city else q
        ans = qag.answer(q_in)  # FIX: không truyền city như tham số thứ 2 nữa
        return JSONResponse(safe_json(ans))
    except Exception:
        log.exception("Ask endpoint error")
        return JSONResponse({"detail": "Internal error while answering"}, status_code=500)


# ---------- (Tuỳ chọn) Admin: reload models sau khi cron xong ----------
@app.post("/_admin/reload")
def admin_reload(x_admin_token: str = Header(default="")):
    """
    Gọi để reload Recommender/QAGuide mà không cần restart server.
    Yêu cầu header: X-Admin-Token = ADMIN_TOKEN trong .env
    """
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    global rec, qag
    try:
        rec = Recommender()  # re-init đọc models *.npy mới
        if QAGuide is not None:
            try:
                qag = QAGuide(encoder=rec.nn)
            except Exception:
                log.exception("QAGuide reload failed; keep None")
                qag = None
        return {"ok": True, "recommender": rec is not None, "qag": qag is not None}
    except Exception:
        log.exception("Reload failed")
        return JSONResponse({"detail": "Reload failed"}, status_code=500)
