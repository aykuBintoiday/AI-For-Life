# src/recommender.py
from __future__ import annotations

import os
import math
from typing import Optional, Iterable

import numpy as np
import pandas as pd

from .etl import load_dataset, strip_accents, parse_closed_days
from .vectorizer import load_tfidf
from .textnorm import normalize_query

from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import CrossEncoder  # optional
except Exception:
    CrossEncoder = None


# -------------------------
# Intent aliases (ý định)
# -------------------------
INTENT_ALIASES = {
    "ba na": [
        "ba na", "ba na hills", "sun world ba na",
        "sunworld bana", "bana", "bà nà", "bà nà hills",
        "sun world bà nà"
    ],
    "asia park": [
        "asia park", "danang wonders", "sun world danang wonders",
        "công viên châu á", "cong vien chau a", "sun wheel"
    ],
    "cau rong": [
        "cầu rồng", "cau rong", "dragon bridge"
    ],
    "my khe": [
        "mỹ khê", "my khe", "my khe beach", "bãi biển mỹ khê"
    ],
    # Bạn bổ sung các cụm khác tại đây nếu cần
}


# -------------------------
# JSON-safe helpers
# -------------------------
def _clean_scalar(v):
    try:
        if v is None:
            return None
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return None
            return float(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        return v
    except Exception:
        return None


def _clean_dict(d: dict | None):
    if not d:
        return None
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _clean_dict(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [_clean_scalar(x) for x in v]
        else:
            out[k] = _clean_scalar(v)
    return out


# -------------------------
# Utility
# -------------------------
def haversine(lat1, lon1, lat2, lon2) -> float:
    try:
        lat1 = float(lat1); lon1 = float(lon1)
        lat2 = float(lat2); lon2 = float(lon2)
    except Exception:
        return 0.0
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def within_open(r, minute_of_day: int, weekday: int) -> bool:
    """True nếu mở cửa tại thời điểm và không nghỉ ngày đó."""
    try:
        om = r.get("open_min", None)
        cm = r.get("close_min", None)
    except Exception:
        om = r["open_min"] if "open_min" in r else None
        cm = r["close_min"] if "close_min" in r else None

    if pd.isna(om) or pd.isna(cm):
        return True

    cs = r.get("closed_set", set())
    if isinstance(cs, (list, tuple)):
        cs = set(int(x) for x in cs if pd.notna(x))
    elif not isinstance(cs, set):
        cs = parse_closed_days(cs) if isinstance(cs, str) else set()

    if weekday in cs:
        return False
    return (om <= minute_of_day <= cm)


def pick(r: dict | None) -> dict | None:
    if not r:
        return None
    keep = [
        "id", "name", "city", "category", "address",
        "open_time", "close_time", "closed_days",
        "duration_min", "avg_cost", "price_level",
        "rating", "review_count", "best_time",
        "desc", "tags", "image_url", "lat", "lon",
        "nn_cos", "ce_score"
    ]
    raw = {k: r.get(k) for k in keep if k in r}
    return _clean_dict(raw)


# -------------------------
# Recommender
# -------------------------
class Recommender:
    """
    - Bi-encoder embeddings → nn_cos cho toàn bộ items.
    - (Tuỳ chọn) Cross-encoder rerank top-K → ce_score.
    - Lọc cứng/ưu tiên theo Ý Định (Bà Nà, Asia Park, …).
    - Không vượt ngân sách/ngày; nếu ≥2 ngày: 1 khách sạn cho cả chuyến.
    """

    def __init__(
        self,
        csv_path: str = "data/itinerary_dataset.csv",
        vec_path: str = "models/tfidf_vectorizer.joblib",
        mat_path: str = "models/tfidf_matrix.npz",
        emb_path: str = "models/nn_embeds.npy",
        emb_meta_path: str = "models/nn_meta.json",
    ):
        # ----- Dữ liệu -----
        try:
            self.df = pd.read_parquet("data/itinerary_dataset.clean.parquet")
        except Exception:
            self.df = load_dataset(csv_path)
        self.df = self.df.reset_index(drop=True)

        # closed_set chuẩn
        def _norm_closed(v):
            if isinstance(v, set):
                return v
            if isinstance(v, (list, tuple)):
                out = set()
                for x in v:
                    try:
                        if pd.notna(x):
                            out.add(int(x))
                    except Exception:
                        pass
                return out
            if isinstance(v, str):
                return parse_closed_days(v)
            return set()
        if "closed_set" in self.df.columns:
            self.df["closed_set"] = self.df["closed_set"].apply(_norm_closed)
        else:
            self.df["closed_set"] = [set()] * len(self.df)

        # ----- TF-IDF (fallback/debug) -----
        try:
            self.vec, self.X = load_tfidf(vec_path, mat_path)
        except Exception:
            self.vec, self.X = None, None
        try:
            self.vocab = set(self.vec.get_feature_names_out()) if self.vec else set()
        except Exception:
            self.vocab = set()

        # ----- Bi-encoder embeddings (bắt buộc) -----
        if not os.path.exists(emb_path):
            raise RuntimeError("Thiếu embeddings DL. Hãy chạy: python build_nn.py")
        self.E = np.load(emb_path)
        if self.E.shape[0] != len(self.df):
            raise RuntimeError(
                f"Embedding rows ({self.E.shape[0]}) != DF rows ({len(self.df)}). Hãy rebuild: python build_nn.py"
            )

        # Model để encode query runtime
        model_name = os.getenv("NN_MODEL", "")
        if not model_name:
            try:
                import json
                with open(emb_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                model_name = meta.get("model", "")
            except Exception:
                model_name = ""
        if not model_name:
            model_name = "intfloat/multilingual-e5-base"
        self.nn = SentenceTransformer(model_name, device="cpu")

        # ----- Cross-Encoder (tuỳ chọn) -----
        self.ce = None
        ce_name = os.getenv("CE_MODEL", "").strip()
        if ce_name:
            if CrossEncoder is None:
                raise RuntimeError("CE_MODEL đã set nhưng chưa có CrossEncoder.")
            self.ce = CrossEncoder(ce_name, device="cpu")

        # ----- Thiết lập khác -----
        self.diversity_cos = float(os.getenv("DIVERSITY_COS", "0.90"))
        self.topk_ce = int(os.getenv("TOPK_CE", "60"))

        # Tâm toạ độ an toàn
        def _safe_center(col: str, default: float) -> float:
            try:
                ser = pd.to_numeric(self.df.get(col), errors="coerce")
                arr = ser.to_numpy(dtype="float64")
                if np.isfinite(arr).any():
                    m = float(np.nanmedian(arr))
                    return m if np.isfinite(m) else default
                return default
            except Exception:
                return default

        self.center_lat = _safe_center("lat", 16.0471)   # Đà Nẵng
        self.center_lon = _safe_center("lon", 108.2068)

    # -------------------------
    # Helpers
    # -------------------------
    def get_place(self, name: str):
        key = strip_accents(name).lower().strip()
        hit = self.df[self.df["norm_name"].astype(str).str.contains(key, na=False)]
        if hit.empty:
            return None
        return pick(hit.iloc[0].to_dict())

    def _encode_query(self, query_text: str) -> tuple[str, np.ndarray]:
        q_norm = normalize_query(query_text, self.vocab)
        q_emb = self.nn.encode(
            [q_norm],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0].astype("float32")
        return q_norm, q_emb

    def _bi_scores(self, q_emb: np.ndarray) -> np.ndarray:
        return self.E @ q_emb  # (N,)

    def _ce_rerank(self, query_text: str, cand_df: pd.DataFrame) -> pd.DataFrame:
        if self.ce is None or cand_df.empty:
            out = cand_df.copy()
            out["ce_score"] = out.get("nn_cos", 0.0)
            return out
        docs = (
            cand_df["name"].astype(str) + ". " +
            cand_df.get("desc", "").astype(str) + " Tags: " +
            cand_df.get("tags", "").astype(str) + " City: " +
            cand_df.get("city", "").astype(str)
        ).tolist()
        pairs = [(query_text, d) for d in docs]
        scores = self.ce.predict(pairs, convert_to_numpy=True)
        out = cand_df.copy()
        out["ce_score"] = scores
        out = out.sort_values("ce_score", ascending=False)
        return out

    def _passes_diversity(self, cand_row_idx: int, used_row_indices: Iterable[int]) -> bool:
        used_row_indices = list(used_row_indices)
        if not used_row_indices:
            return True
        v = self.E[cand_row_idx]
        U = self.E[used_row_indices]
        sims = U @ v
        return float(np.max(sims)) <= self.diversity_cos

    # ---------- Ý Định ----------
    def _extract_intent(self, q_norm: str) -> list[str]:
        keys = []
        for k, aliases in INTENT_ALIASES.items():
            if any(a in q_norm for a in aliases):
                keys.append(k)
        return keys

    def _filter_or_boost_by_intent(self, df: pd.DataFrame, q_norm: str) -> pd.DataFrame:
        intents = self._extract_intent(q_norm)
        if not intents or df.empty:
            return df

        def has_alias(row) -> bool:
            pool = " ".join([
                str(row.get("name", "")),
                str(row.get("tags", "")),
                str(row.get("desc", "")),
            ])
            pool = strip_accents(pool).lower()
            for k in intents:
                for a in INTENT_ALIASES.get(k, []):
                    if a in pool:
                        return True
            return False

        m = df.apply(has_alias, axis=1)

        # Nếu có đủ ứng viên khớp ý định → LỌC CỨNG
        if int(m.sum()) >= 3:
            return df[m].copy()

        # Nếu ít → BOOST điểm để ưu tiên nhưng vẫn giữ phần còn lại
        out = df.copy()
        bonus = 0.18
        out.loc[m, "nn_cos"] = out.loc[m, "nn_cos"] + bonus
        if "ce_score" in out.columns:
            out.loc[m, "ce_score"] = out.loc[m, "ce_score"] + bonus
        return out

    # ---------- Chọn Slot ----------
    def _pick_slot(
        self,
        cand: pd.DataFrame,
        slot_minute: int,
        weekday: int,
        anchor_lat: float,
        anchor_lon: float,
        budget_allowed: float,
        used_ids: set[int],
        used_rows: set[int],
        min_close: Optional[int] = None,
    ) -> dict | None:
        # loại đã dùng
        cand = cand[~cand["id"].isin(used_ids)].copy()
        if cand.empty:
            return None

        # khoảng cách
        cand["dist"] = [
            haversine(anchor_lat, anchor_lon, la or 0.0, lo or 0.0)
            for la, lo in zip(cand.get("lat", 0), cand.get("lon", 0))
        ]

        # mở cửa
        def ok_time(row) -> bool:
            if not within_open(row, slot_minute, weekday):
                return False
            if min_close is not None:
                cm = row.get("close_min", np.nan)
                if not (pd.isna(cm) or cm >= min_close):
                    return False
            return True

        c1 = cand[cand.apply(ok_time, axis=1)]
        if c1.empty:
            c1 = cand

        # ngân sách slot
        c1["cost"] = pd.to_numeric(c1.get("avg_cost", 0), errors="coerce").fillna(0.0)
        under = c1[c1["cost"] <= budget_allowed]
        if under.empty:
            soft = c1[c1["cost"] <= budget_allowed * 1.10]  # cho phép vượt nhẹ slot 10%
            base = soft if not soft.empty else c1
        else:
            base = under

        # xếp theo: ce_score (nếu có) → nn_cos → dist (gần) → |cost - budget_allowed| (ổn định chi phí)
        if "ce_score" in base.columns:
            sort_cols = ["ce_score", "nn_cos", "dist", "cost"]
            ascending = [False, False, True, True]
        else:
            sort_cols = ["nn_cos", "dist", "cost"]
            ascending = [False, True, True]

        base = base.sort_values(by=sort_cols, ascending=ascending)

        # đa dạng
        for _, r in base.iterrows():
            row_idx = int(r["row_index"])
            if self._passes_diversity(row_idx, used_rows):
                return r.to_dict()

        return base.iloc[0].to_dict() if not base.empty else None

    def _pick_hotel_once(
        self,
        H: pd.DataFrame,
        per_day_budget: float,
        used_ids: set[int],
        used_rows: set[int],
        center_lat: float,
        center_lon: float,
    ) -> dict | None:
        h = H[~H["id"].isin(used_ids)].copy()
        if h.empty:
            return None

        h["dist"] = [
            haversine(center_lat, center_lon, la or 0.0, lo or 0.0)
            for la, lo in zip(h.get("lat", 0), h.get("lon", 0))
        ]
        h["cost"] = pd.to_numeric(h.get("avg_cost", 0), errors="coerce").fillna(0.0)

        # KS không quá 70% ngân sách/ngày (có thể chỉnh)
        under = h[h["cost"] <= per_day_budget * 0.70]
        if under.empty:
            under = h[h["cost"] <= per_day_budget * 1.00]
        base = under if not under.empty else h

        # xếp KS: ce_score → nn_cos → dist → |cost - 0.55*per_day|
        target = per_day_budget * 0.55
        base["gap"] = (base["cost"] - target).abs()

        if "ce_score" in base.columns:
            sort_cols = ["ce_score", "nn_cos", "gap", "dist"]
            ascending = [False, False, True, True]
        else:
            sort_cols = ["nn_cos", "gap", "dist"]
            ascending = [False, True, True]

        base = base.sort_values(by=sort_cols, ascending=ascending)

        for _, r in base.iterrows():
            row_idx = int(r["row_index"])
            if self._passes_diversity(row_idx, used_rows):
                return r.to_dict()

        return base.iloc[0].to_dict()

    # -------------------------
    # Itinerary
    # -------------------------
    def itinerary(self, query_text: str, days: int, budget_total: int, city: Optional[str] = None):
        # 1) Lọc city
        df = self.df.copy()
        if city:
            key = strip_accents(city).lower()
            df = df[df["city"].astype(str).str.lower().apply(strip_accents).str.contains(key, na=False)]
        if df.empty:
            df = self.df.copy()

        df["row_index"] = df.index

        # 2) Embedding điểm
        q_norm, q_emb = self._encode_query(query_text)
        nn_cos_all = self._bi_scores(q_emb)
        df = df.assign(nn_cos=nn_cos_all[df.index])

        # 3) Ý định: lọc/boost
        df = self._filter_or_boost_by_intent(df, q_norm)

        # 4) CE rerank theo nhóm
        def rerank_block(block: pd.DataFrame) -> pd.DataFrame:
            if block.empty:
                return block
            sub = block.sort_values("nn_cos", ascending=False).head(self.topk_ce).copy()
            sub = self._ce_rerank(query_text, sub)
            return sub

        S = rerank_block(df[df["category"] == "sightseeing"].copy())
        F = rerank_block(df[df["category"] == "food"].copy())
        H = rerank_block(df[df["category"] == "hotel"].copy())
        Evt = rerank_block(df[df["category"] == "event"].copy())

        # 5) Một khách sạn cho cả chuyến (nếu có từ 1 ngày trở lên)
        per_day_budget = max(0, int(budget_total)) // max(1, int(days))
        used_ids: set[int] = set()
        used_rows: set[int] = set()

        hotel = self._pick_hotel_once(
            H=H, per_day_budget=per_day_budget,
            used_ids=used_ids, used_rows=used_rows,
            center_lat=self.center_lat, center_lon=self.center_lon
        )
        if hotel:
            used_ids.add(hotel["id"]); used_rows.add(int(hotel["row_index"]))
            hotel_cost = float(pd.to_numeric(hotel.get("avg_cost", 0), errors="coerce") or 0.0)
        else:
            hotel_cost = 0.0

        # chi phí KS quy ra mỗi ngày (để không vượt ngân sách/ngày)
        hotel_per_day = hotel_cost

        # 6) Kế hoạch từng ngày
        plan = []
        slot_share = {"morning": 0.25, "lunch": 0.15, "afternoon": 0.30, "evening": 0.30}

        for d in range(days):
            wd = d % 7
            anchor_lat = hotel.get("lat", self.center_lat) if hotel else self.center_lat
            anchor_lon = hotel.get("lon", self.center_lon) if hotel else self.center_lon

            # Ngân sách còn lại trong ngày sau khi tính KS
            budget_left = max(0.0, float(per_day_budget) - hotel_per_day)

            # MORNING
            morning = self._pick_slot(
                cand=S, slot_minute=9 * 60, weekday=wd,
                anchor_lat=anchor_lat, anchor_lon=anchor_lon,
                budget_allowed=budget_left * slot_share["morning"],
                used_ids=used_ids, used_rows=used_rows
            )
            if morning:
                cost_m = float(pd.to_numeric(morning.get("avg_cost", 0), errors="coerce") or 0.0)
                budget_left = max(0.0, budget_left - cost_m)
                used_ids.add(morning["id"]); used_rows.add(int(morning["row_index"]))
                anchor_lat = morning.get("lat", anchor_lat); anchor_lon = morning.get("lon", anchor_lon)

            # LUNCH
            lunch = self._pick_slot(
                cand=F, slot_minute=11 * 60 + 30, weekday=wd,
                anchor_lat=anchor_lat, anchor_lon=anchor_lon,
                budget_allowed=budget_left * slot_share["lunch"],
                used_ids=used_ids, used_rows=used_rows
            )
            if lunch:
                cost_l = float(pd.to_numeric(lunch.get("avg_cost", 0), errors="coerce") or 0.0)
                budget_left = max(0.0, budget_left - cost_l)
                used_ids.add(lunch["id"]); used_rows.add(int(lunch["row_index"]))
                anchor_lat = lunch.get("lat", anchor_lat); anchor_lon = lunch.get("lon", anchor_lon)

            # AFTERNOON (đóng tối thiểu 16:30)
            afternoon = self._pick_slot(
                cand=S, slot_minute=15 * 60, weekday=wd,
                anchor_lat=anchor_lat, anchor_lon=anchor_lon,
                budget_allowed=budget_left * slot_share["afternoon"],
                used_ids=used_ids, used_rows=used_rows,
                min_close=16 * 60 + 30
            )
            if afternoon:
                cost_a = float(pd.to_numeric(afternoon.get("avg_cost", 0), errors="coerce") or 0.0)
                budget_left = max(0.0, budget_left - cost_a)
                used_ids.add(afternoon["id"]); used_rows.add(int(afternoon["row_index"]))
                anchor_lat = afternoon.get("lat", anchor_lat); anchor_lon = afternoon.get("lon", anchor_lon)

            # EVENING (đóng tối thiểu 21:30)
            evening = self._pick_slot(
                cand=Evt, slot_minute=19 * 60, weekday=wd,
                anchor_lat=anchor_lat, anchor_lon=anchor_lon,
                budget_allowed=budget_left * slot_share["evening"],
                used_ids=used_ids, used_rows=used_rows,
                min_close=21 * 60 + 30
            )
            if evening:
                cost_e = float(pd.to_numeric(evening.get("avg_cost", 0), errors="coerce") or 0.0)
                budget_left = max(0.0, budget_left - cost_e)
                used_ids.add(evening["id"]); used_rows.add(int(evening["row_index"]))

            # Tổng chi phí ngày (không vượt per_day_budget)
            def _cost(x):
                return float(pd.to_numeric((x or {}).get("avg_cost", 0), errors="coerce") or 0.0)

            day_cost = hotel_per_day + _cost(morning) + _cost(lunch) + _cost(afternoon) + _cost(evening)
            over_budget = bool(day_cost > per_day_budget * 1.02)  # nhẹ 2% để tránh làm tròn

            plan.append({
                "day": d + 1,
                "morning": pick(morning),
                "lunch": pick(lunch),
                "afternoon": pick(afternoon),
                "evening": pick(evening),
                "hotel": pick(hotel),
                "day_cost_estimate": int(round(day_cost)),
                "over_budget": over_budget,
            })

        total = int(sum(int(x["day_cost_estimate"]) for x in plan))
        result = {
            "days": int(days),
            "per_day_budget": int(per_day_budget),
            "total_cost_estimate": int(total),
            "itinerary": [
                _clean_dict({
                    "day": int(x["day"]),
                    "morning": x["morning"],
                    "lunch": x["lunch"],
                    "afternoon": x["afternoon"],
                    "evening": x["evening"],
                    "hotel": x["hotel"],
                    "day_cost_estimate": int(x["day_cost_estimate"]),
                    "over_budget": bool(x["over_budget"]),
                }) for x in plan
            ],
        }
        return result
