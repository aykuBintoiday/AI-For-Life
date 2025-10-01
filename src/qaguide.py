# src/qaguide.py
from __future__ import annotations
import json, re, unicodedata
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

# -------- helpers ----------
def strip_accents(s: str) -> str:
    s = str(s or "")
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

def norm(s: str) -> str:
    s = strip_accents(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

QUESTION_KEYS = [
    r"m(á|a)y gi(ờ|o)", r"gi(ờ|o)\s*m(ở|o)", r"gi(ờ|o)\s*đ(ó|o)ng",
    r"m(ở|o)\s*c(ửa|ua)", r"đ(ó|o)ng\s*c(ửa|ua)", r"l(à|a)\s*g(ì|i)",
    r"ở\s*đ(â|a)u", r"bao\s*nhi(ê|e)u", r"gi(á|a)", r"history", r"l(ị|i)ch\s*s(ử|u)"
]
Q_RE = re.compile("|".join(QUESTION_KEYS), re.I)

CITY_MAP = [
    ("đà nẵng", "Da Nang"), ("da nang", "Da Nang"), ("danang", "Da Nang"),
    ("hội an", "Hoi An"), ("hoi an", "Hoi An"),
    ("huế", "Hue"), ("hue", "Hue"),
]

class QAGuide:
    def __init__(
        self,
        encoder=None,
        parquet_path: str = "data/qna_places.clean.parquet",
        embeds_path: str = "models/qna_embeds.npy",
        meta_path: str = "models/qna_meta.json",
    ):
        self.encoder = encoder

        p = Path(parquet_path)
        if not p.exists():
            raise RuntimeError(f"Không thấy {parquet_path}. Hãy chạy: python -m src.etl_qna data/qna_places.csv")
        self.df = pd.read_parquet(p)

        # ensure columns exist
        for c in ["norm_name", "text", "alias_norms", "open_time", "close_time", "best_time", "address", "city", "desc"]:
            if c not in self.df.columns:
                self.df[c] = ""

        # alias_norms => list
        def as_list(x):
            if isinstance(x, list):
                return [str(v) for v in x]
            if isinstance(x, str):
                xs = x.strip()
                if xs.startswith("["):
                    try:
                        return [str(v) for v in json.loads(xs)]
                    except Exception:
                        pass
                return [t.strip() for t in xs.split(";") if t.strip()]
            return []
        self.df["alias_norms"] = self.df["alias_norms"].apply(as_list)

        self.df["norm_name"] = self.df["norm_name"].map(norm)
        self.df["city_norm"] = self.df["city"].map(norm)
        self.names_arr: List[str] = self.df["norm_name"].astype(str).tolist()
        self.aliases_arr: List[List[str]] = self.df["alias_norms"].tolist()

        # optional embeddings
        self.embeds = None
        ne = Path(embeds_path)
        if ne.exists():
            try:
                self.embeds = np.load(ne)
            except Exception:
                self.embeds = None

    # ---------- scoring ----------
    def _encode(self, q: str) -> Optional[np.ndarray]:
        if self.encoder is None:
            return None
        try:
            v = self.encoder.encode([q], normalize_embeddings=True)
            return v[0]
        except Exception:
            return None

    def _sim(self, qv: np.ndarray) -> Optional[np.ndarray]:
        if qv is None or self.embeds is None:
            return None
        # embeddings đã normalize => dot = cosine
        return self.embeds @ qv

    def _lexical_scores(self, q_norm: str) -> np.ndarray:
        toks = [t for t in q_norm.split() if len(t) >= 2]
        scores = np.zeros(len(self.df), dtype=np.float32)
        for i, (name, aliases) in enumerate(zip(self.names_arr, self.aliases_arr)):
            hay = " ".join([name] + aliases)
            sc = 0.0
            for t in toks:
                if t in hay:
                    sc += 1.0
            if name and name in q_norm:
                sc += 1.5  # bonus nếu khớp tên đầy đủ
            scores[i] = sc
        if scores.max() > 0:
            scores = scores / (scores.max() + 1e-9)
        return scores

    def _city_bonus(self, city: Optional[str]) -> Optional[np.ndarray]:
        if not city:
            return None
        c = norm(city)
        if not c:
            return None
        return self.df["city_norm"].str.contains(c, na=False).values.astype(np.float32) * 0.2

    def search(self, q: str, city: Optional[str] = None, k: int = 5) -> List[int]:
        qn = norm(q)
        if not qn:
            return []
        lex = self._lexical_scores(qn)
        emb = None
        qv = self._encode(qn)
        if qv is not None and self.embeds is not None:
            emb = self._sim(qv)

        if emb is None:
            score = lex
        else:
            # câu ngắn ưu tiên lexical hơn
            w_lex = 0.7 if len(qn) <= 16 else 0.35
            score = w_lex * lex + (1.0 - w_lex) * emb

        bonus = self._city_bonus(city)
        if bonus is not None:
            score = score + bonus

        order = np.argsort(-score)
        top = [int(i) for i in order[:k] if score[i] > 0]
        return top

    # ---------- public ----------
    def answer(self, q: str) -> Dict[str, Any]:
        text_q = str(q or "")
        # cố gắng đoán city từ câu
        city = None
        nq = norm(text_q)
        for key, cval in CITY_MAP:
            if key in nq:
                city = cval
                break

        hits = self.search(text_q, city=city, k=5)
        if not hits:
            return {"text": "Không tìm thấy thông tin phù hợp."}

        i = hits[0]
        row = self.df.iloc[i]

        name = row.get("name", "")
        address = row.get("address", "")
        open_time = (row.get("open_time", "") or "").strip()
        close_time = (row.get("close_time", "") or "").strip()
        best_time = (row.get("best_time", "") or "").strip()
        desc = (row.get("desc", "") or "").strip()
        city_disp = (row.get("city", "") or "").strip()

        # Nếu câu kiểu hỏi giờ/ở đâu/giá... thì ưu tiên câu súc tích
        concise = None
        if Q_RE.search(text_q):
            concise = f"{name}: mở {open_time or '—'} — đóng {close_time or '—'}."

        html = f"""
<div style="border:1px solid #1e293b;border-radius:12px;padding:12px">
  <div style="font-weight:700;font-size:16px;margin-bottom:6px">{name}</div>
  <div style="font-size:13px;color:#9aa4b2;margin-bottom:8px">
    {(address or '—')}{(' · ' + city_disp) if city_disp else ''}
  </div>
  <div style="display:grid;grid-template-columns:120px 1fr;gap:6px;font-size:14px">
    <div>Giờ mở cửa</div><div>{open_time or '—'} – {close_time or '—'}</div>
    <div>Thời điểm đẹp</div><div>{best_time or '—'}</div>
  </div>
  <div style="margin-top:8px;line-height:1.5">{desc}</div>
</div>
""".strip()

        return {
            "text": concise,
            "html": html,
            "place": {"id": int(row.get("id", 0)), "name": name},
        }
