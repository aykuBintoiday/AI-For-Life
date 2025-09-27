# src/recommender.py
from __future__ import annotations

import math
import os
import json
from typing import Optional, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity  # fallback/debug

from .etl import load_dataset, strip_accents, parse_closed_days
from .vectorizer import load_tfidf
from .textnorm import normalize_query

# ====== Deep Learning encoders ======
from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import CrossEncoder  # optional
except Exception:
    CrossEncoder = None


# -------------------------
# Utility helpers
# -------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Khoảng cách đường tròn lớn (km). An toàn với None/NaN."""
    try:
        lat1 = float(lat1) if pd.notna(lat1) else 0.0
        lon1 = float(lon1) if pd.notna(lon1) else 0.0
        lat2 = float(lat2) if pd.notna(lat2) else 0.0
        lon2 = float(lon2) if pd.notna(lon2) else 0.0
    except Exception:
        return 0.0
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def within_open(r, minute_of_day: int, weekday: int) -> bool:
    """
    True nếu địa điểm mở cửa tại minute_of_day và không thuộc closed_days.
    Thiếu dữ liệu → coi như mở (True).
    """
    try:
        om = r.get("open_min", None)
        cm = r.get("close_min", None)
    except AttributeError:
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
    return (int(om) <= int(minute_of_day) <= int(cm))


def pick(r: dict | None) -> dict | None:
    """Trích các field cần trả về UI."""
    if not r:
        return None
    keep = [
        "id", "name", "city", "category", "address",
        "open_time", "close_time", "closed_days",
        "duration_min", "avg_cost", "price_level",
        "rating", "review_count", "best_time",
        "desc", "tags", "image_url", "lat", "lon",
        # debug scores (ẩn nếu không dùng)
        "nn_cos", "ce_score"
    ]
    return {k: r.get(k) for k in keep if k in r}


# -------------------------
# Recommender (DL-first)
# -------------------------
class Recommender:
    """
    Xếp hạng thuần DL:
    - Bi-encoder (embeddings) → điểm 'nn_cos' duy nhất cho toàn bộ items.
    - (Tuỳ chọn) Cross-encoder rerank top-K → điểm 'ce_score' dùng xếp hạng cuối.
    - Giữ filter cứng (giờ/Ngày nghỉ/Ngân sách/Khoảng cách).
    - Đa dạng theo ngưỡng tương đồng (skip nếu cosine với item đã chọn > threshold).
    """

    def __init__(
        self,
        csv_path: str = "data/itinerary_dataset.csv",
        vec_path: str = "models/tfidf_vectorizer.joblib",
        mat_path: str = "models/tfidf_matrix.npz",
        emb_path: str = "models/nn_embeds.npy",
        emb_meta_path: str = "models/nn_meta.json",
    ):
        # ===== Load dữ liệu =====
        try:
            self.df = pd.read_parquet("data/itinerary_dataset.clean.parquet")
        except Exception:
            self.df = load_dataset(csv_path)
        self.df = self.df.reset_index(drop=True)

        # Chuẩn hoá closed_set
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

        # ===== TF-IDF (fallback/debug) =====
        try:
            self.vec, self.X = load_tfidf(vec_path, mat_path)
        except Exception:
            self.vec, self.X = None, None

        # vocab cho normalize_query (sửa lỗi gõ nhẹ)
        try:
            self.vocab = set(self.vec.get_feature_names_out()) if self.vec else set()
        except Exception:
            self.vocab = set()

        # ===== Embeddings bi-encoder (bắt buộc) =====
        if not os.path.exists(emb_path):
            raise RuntimeError(
                "Thiếu embeddings DL. Hãy chạy: python build_nn.py"
            )
        self.E = np.load(emb_path)  # (N, D), đã normalize (L2) trong build_nn.py
        if self.E.shape[0] != len(self.df):
            raise RuntimeError(
                f"Embedding rows ({self.E.shape[0]}) != DF rows ({len(self.df)}). Hãy rebuild: python build_nn.py"
            )

        # Load model bi-encoder để encode query runtime
        model_name = os.getenv("NN_MODEL", "")
        if not model_name:
            try:
                with open(emb_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                model_name = meta.get("model", "")
            except Exception:
                model_name = ""
        if not model_name:
            model_name = "intfloat/multilingual-e5-base"
        self.nn = SentenceTransformer(model_name, device="cpu")

        # ===== Tuỳ chọn Cross-Encoder reranker =====
        self.ce = None
        ce_name = os.getenv("CE_MODEL", "").strip()
        if ce_name:
            if CrossEncoder is None:
                raise RuntimeError("Đã set CE_MODEL nhưng thiếu CrossEncoder trong sentence-transformers.")
            self.ce = CrossEncoder(ce_name, device="cpu")

        # ===== Thiết lập khác =====
        self.diversity_cos = float(os.getenv("DIVERSITY_COS", "0.90"))   # ngưỡng loại trùng lặp
        self.topk_ce = int(os.getenv("TOPK_CE", "60"))                    # số lượng top-K đưa vào CE
        # Tâm toạ độ (bỏ các (0,0))
        df_valid = self.df[(self.df["lat"].notna()) & (self.df["lon"].notna()) & (self.df["lat"] != 0.0) & (self.df["lon"] != 0.0)]
        if not df_valid.empty:
            self.center_lat, self.center_lon = df_valid["lat"].median(), df_valid["lon"].median()
        else:
            self.center_lat, self.center_lon = 16.0471, 108.2068  # Đà Nẵng fallback

    # -------------------------
    # Public helpers
    # -------------------------
    def get_place(self, name: str):
        key = strip_accents(name).lower().strip()
        hit = self.df[self.df["norm_name"].str.contains(key, na=False)]
        if hit.empty:
            return None
        return pick(hit.iloc[0].to_dict())

    # -------------------------
    # Core retrieval
    # -------------------------
    def _encode_query(self, query_text: str) -> tuple[str, np.ndarray]:
        """Chuẩn hoá truy vấn + sinh embedding (L2 normed)."""
        q_norm = normalize_query(query_text, self.vocab)
        q_emb = self.nn.encode(
            [q_norm],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0].astype("float32")
        return q_norm, q_emb

    def _bi_scores(self, q_emb: np.ndarray) -> np.ndarray:
        """Cosine = dot (vì E đã L2-normalized)."""
        return self.E @ q_emb  # (N,)

    def _ce_rerank(self, query_text: str, cand_df: pd.DataFrame) -> pd.DataFrame:
        """Rerank top-K bằng Cross-Encoder (nếu bật). Trả về cand_df có cột 'ce_score' và đã sort."""
        if self.ce is None or cand_df.empty:
            out = cand_df.copy()
            if "ce_score" not in out.columns:
                out["ce_score"] = out["nn_cos"]
            return out
        # Dùng text giàu ngữ nghĩa cho CE
        docs = (cand_df["name"].astype(str) + ". " +
                cand_df["desc"].astype(str) + " Tags: " +
                cand_df["tags"].astype(str) + " City: " +
                cand_df["city"].astype(str)).tolist()
        pairs = [(query_text, d) for d in docs]
        scores = self.ce.predict(pairs, convert_to_numpy=True)
        out = cand_df.copy()
        out["ce_score"] = scores
        out = out.sort_values("ce_score", ascending=False)
        return out

    def _passes_diversity(self, row_idx: int | None, used_row_indices: Iterable[int]) -> bool:
        """Loại nếu cosine(cand, any selected) > threshold."""
        if row_idx is None or not used_row_indices:
            return True
        v = self.E[row_idx]  # (D,)
        U = self.E[list(used_row_indices)]  # (k, D)
        if U.size == 0:
            return True
        sims = U @ v  # (k,)
        return float(np.max(sims)) <= self.diversity_cos

    # -------------------------
    # Itinerary
    # -------------------------
    def itinerary(self, query_text: str, days: int, budget_total: int, city: Optional[str] = None):
        # 1) Lọc theo city (nếu có)
        df = self.df.copy()
        if city:
            key = strip_accents(city).lower()
            df = df[df["city"].str.lower().apply(strip_accents).str.contains(key, na=False)]
        if df.empty:
            df = self.df.copy()

        # 2) Bi-encoder: điểm nn_cos cho toàn bộ items
        q_norm, q_emb = self._encode_query(query_text)
        nn_cos_all = self._bi_scores(q_emb)  # (N,)
        df = df.assign(nn_cos=nn_cos_all[df.index])

        # 3) (Tuỳ chọn) CE rerank từng nhóm category từ top-K bi-encoder
        def rerank_block(block: pd.DataFrame) -> pd.DataFrame:
            if block.empty:
                return block
            sub = block.sort_values("nn_cos", ascending=False).head(self.topk_ce).copy()
            sub = self._ce_rerank(query_text, sub)
            # đảm bảo có ce_score
            if "ce_score" not in sub.columns:
                sub["ce_score"] = sub["nn_cos"]
            return sub

        S = rerank_block(df[df["category"] == "sightseeing"].copy())
        F = rerank_block(df[df["category"] == "food"].copy())
        H = rerank_block(df[df["category"] == "hotel"].copy())
        Evt = rerank_block(df[df["category"] == "event"].copy())

        # 4) Chọn theo slot thời gian (không trộn điểm thủ công)
        per_day_budget = max(0, budget_total) // max(1, days)
        used_ids: set[int] = set()
        plan = []

        # Map id -> row index (để check diversity qua embeddings)
        id_series = self.df["id"].astype(int)
        id2row = {int(v): i for i, v in enumerate(id_series.tolist())}
        used_rows: set[int] = set()

        for d in range(days):
            wd = d % 7  # 0=Mon

            # ========== HOTEL ==========
            target_hotel = per_day_budget * 0.55
            h = H[~H["id"].isin(used_ids)].copy()
            if not h.empty:
                h["cost_gap"] = (h["avg_cost"] - target_hotel).abs()
                h["dist"] = [
                    haversine(self.center_lat, self.center_lon, la, lo)
                    for la, lo in zip(h["lat"], h["lon"])
                ]
                # ưu tiên: ce_score ↓, nn_cos ↓, cost_gap ↑, dist ↑
                h = h.sort_values(by=["ce_score", "nn_cos", "cost_gap", "dist"],
                                  ascending=[False, False, True, True])
            hotel = h.iloc[0].to_dict() if not h.empty else None
            if hotel:
                used_ids.add(hotel["id"])
                used_rows.add(id2row.get(int(hotel["id"]), -1))
            anchor_lat = (hotel or {}).get("lat", self.center_lat) or self.center_lat
            anchor_lon = (hotel or {}).get("lon", self.center_lon) or self.center_lon

            # ========== MORNING (09:00) ==========
            morning_time = 9 * 60
            s1_all = S[~S["id"].isin(used_ids)].copy()
            if not s1_all.empty:
                s1_all["dist"] = [
                    haversine(anchor_lat, anchor_lon, la, lo)
                    for la, lo in zip(s1_all["lat"], s1_all["lon"])
                ]
                s1_all = s1_all.sort_values(by=["ce_score", "nn_cos", "dist"],
                                            ascending=[False, False, True])
                morning = None
                for _, r in s1_all.iterrows():
                    row_idx = id2row.get(int(r["id"]))
                    if within_open(r, morning_time, wd) and self._passes_diversity(row_idx, used_rows):
                        morning = r.to_dict()
                        break
                if not morning:
                    for _, r in s1_all.iterrows():
                        row_idx = id2row.get(int(r["id"]))
                        if self._passes_diversity(row_idx, used_rows):
                            morning = r.to_dict()
                            break
            else:
                morning = None
            if morning:
                used_ids.add(morning["id"])
                used_rows.add(id2row.get(int(morning["id"]), -1))

            # ========== LUNCH (11:30) ==========
            lunch_time = 11 * 60 + 30
            anchor_lat = (morning or {}).get("lat", anchor_lat)
            anchor_lon = (morning or {}).get("lon", anchor_lon)
            f1_all = F[~F["id"].isin(used_ids)].copy()
            if not f1_all.empty:
                f1_all["dist"] = [
                    haversine(anchor_lat, anchor_lon, la, lo)
                    for la, lo in zip(f1_all["lat"], f1_all["lon"])
                ]
                f1_all = f1_all.sort_values(by=["ce_score", "nn_cos", "dist"],
                                            ascending=[False, False, True])
                lunch = None
                for _, r in f1_all.iterrows():
                    row_idx = id2row.get(int(r["id"]))
                    if within_open(r, lunch_time, wd) and self._passes_diversity(row_idx, used_rows):
                        lunch = r.to_dict()
                        break
                if not lunch:
                    for _, r in f1_all.iterrows():
                        row_idx = id2row.get(int(r["id"]))
                        if self._passes_diversity(row_idx, used_rows):
                            lunch = r.to_dict()
                            break
            else:
                lunch = None
            if lunch:
                used_ids.add(lunch["id"])
                used_rows.add(id2row.get(int(lunch["id"]), -1))

            # ========== AFTERNOON (15:00, cố gắng mở ≥16:30) ==========
            afternoon_time = 15 * 60
            min_close = 16 * 60 + 30
            anchor_lat = (lunch or {}).get("lat", anchor_lat)
            anchor_lon = (lunch or {}).get("lon", anchor_lon)
            s2_all = S[~S["id"].isin(used_ids)].copy()
            if not s2_all.empty:
                s2_all["dist"] = [
                    haversine(anchor_lat, anchor_lon, la, lo)
                    for la, lo in zip(s2_all["lat"], s2_all["lon"])
                ]
                s2_all = s2_all.sort_values(by=["ce_score", "nn_cos", "dist"],
                                            ascending=[False, False, True])
                afternoon = None
                for _, r in s2_all.iterrows():
                    row_idx = id2row.get(int(r["id"]))
                    cm = r.get("close_min", None)
                    ok_close = True if pd.isna(cm) else (cm >= min_close)
                    if within_open(r, afternoon_time, wd) and ok_close and self._passes_diversity(row_idx, used_rows):
                        afternoon = r.to_dict()
                        break
                if not afternoon:
                    for _, r in s2_all.iterrows():
                        row_idx = id2row.get(int(r["id"]))
                        if self._passes_diversity(row_idx, used_rows):
                            afternoon = r.to_dict()
                            break
            else:
                afternoon = None
            if afternoon:
                used_ids.add(afternoon["id"])
                used_rows.add(id2row.get(int(afternoon["id"]), -1))

            # ========== EVENING EVENT (19:00–21:30) ==========
            evening_time = 19 * 60
            min_close_e = 21 * 60 + 30
            anchor_lat_e = (afternoon or {}).get("lat", anchor_lat)
            anchor_lon_e = (afternoon or {}).get("lon", anchor_lon)
            e_all = Evt[~Evt["id"].isin(used_ids)].copy()
            if not e_all.empty:
                e_all["dist"] = [
                    haversine(anchor_lat_e, anchor_lon_e, la, lo)
                    for la, lo in zip(e_all["lat"], e_all["lon"])
                ]
                e_all = e_all.sort_values(by=["ce_score", "nn_cos", "dist"],
                                          ascending=[False, False, True])
                evening = None
                for _, r in e_all.iterrows():
                    row_idx = id2row.get(int(r["id"]))
                    cm = r.get("close_min", None)
                    ok_close = True if pd.isna(cm) else (cm >= min_close_e)
                    if within_open(r, evening_time, wd) and ok_close and self._passes_diversity(row_idx, used_rows):
                        evening = r.to_dict()
                        break
                if not evening:
                    for _, r in e_all.iterrows():
                        row_idx = id2row.get(int(r["id"]))
                        if self._passes_diversity(row_idx, used_rows):
                            evening = r.to_dict()
                            break
            else:
                evening = None
            if evening:
                used_ids.add(evening["id"])
                used_rows.add(id2row.get(int(evening["id"]), -1))

            # ========== Tính tổng chi phí ngày ==========
            day_cost = sum([
                (morning or {}).get("avg_cost", 0),
                (lunch or {}).get("avg_cost", 0),
                (afternoon or {}).get("avg_cost", 0),
                (evening or {}).get("avg_cost", 0),
                (hotel or {}).get("avg_cost", 0),
            ])
            plan.append({
                "day": d + 1,
                "morning": pick(morning),
                "lunch": pick(lunch),
                "afternoon": pick(afternoon),
                "evening": pick(evening),
                "hotel": pick(hotel),
                "day_cost_estimate": int(day_cost),
                "over_budget": day_cost > per_day_budget * 1.1,
            })

        total = sum(x["day_cost_estimate"] for x in plan)
        return {
            "days": days,
            "per_day_budget": int(per_day_budget),
            "total_cost_estimate": int(total),
            "itinerary": plan,
        }
