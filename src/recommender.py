# src/recommender.py
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from .etl import load_dataset, strip_accents, parse_closed_days
from .vectorizer import load_tfidf
from .textnorm import normalize_query


def haversine(lat1, lon1, lat2, lon2):
    """
    Khoảng cách đường tròn lớn (km). An toàn với float.
    """
    try:
        lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    except Exception:
        return 0.0
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon1 - lon2)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def within_open(r, minute_of_day: int, weekday: int) -> bool:
    """
    Kiểm tra giờ mở cửa + ngày nghỉ. Nếu thiếu dữ liệu → cho qua (True).
    """
    try:
        om = r.get("open_min")
        cm = r.get("close_min")
    except Exception:
        om = r["open_min"] if "open_min" in r else None
        cm = r["close_min"] if "close_min" in r else None

    # Thiếu giờ → coi như mở
    if pd.isna(om) or pd.isna(cm):
        return True

    cs = r.get("closed_set", set())
    if isinstance(cs, (list, tuple)):
        cs = set(cs)
    elif not isinstance(cs, set):
        cs = parse_closed_days(cs) if isinstance(cs, str) else set()

    if weekday in cs:
        return False
    return (om <= minute_of_day <= cm)


def pick(r: dict | None) -> dict | None:
    if not r:
        return None
    keep = [
        "id", "name", "city", "category", "address", "open_time", "close_time", "closed_days",
        "duration_min", "avg_cost", "price_level", "rating", "review_count", "best_time",
        "desc", "tags", "image_url", "lat", "lon", "cos"
    ]
    return {k: r.get(k) for k in keep if k in r}


def safe_dist(lat_a, lon_a, la, lo):
    """
    Khoảng cách an toàn: nếu thiếu toạ độ hoặc (0,0) thì không phạt (trả 0).
    """
    try:
        if la is None or lo is None:
            return 0.0
        la, lo = float(la), float(lo)
        if la == 0.0 and lo == 0.0:
            return 0.0
        if lat_a is None or lon_a is None:
            return 0.0
        return haversine(float(lat_a), float(lon_a), la, lo)
    except Exception:
        return 0.0


# Map city “lỏng” để người dùng gõ biến thể vẫn khớp
CITY_MAP = {
    "dn": "da nang",
    "danang": "da nang",
    "đà nẵng": "da nang",
}


class Recommender:
    def __init__(
        self,
        csv_path: str = "data/itinerary_dataset.csv",
        vec_path: str = "models/tfidf_vectorizer.joblib",
        mat_path: str = "models/tfidf_matrix.npz",
    ):
        # Ưu tiên parquet sạch; fallback CSV
        try:
            self.df = pd.read_parquet("data/itinerary_dataset.clean.parquet")
        except Exception:
            self.df = load_dataset(csv_path)

        # Đồng bộ index 0..N-1 cho khớp TF-IDF
        self.df = self.df.reset_index(drop=True)

        # Chuẩn hoá closed_set (phòng khi parquet đọc kiểu lạ)
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

        # Load TF-IDF
        self.vec, self.X = load_tfidf(vec_path, mat_path)

        # vocab cho normalize_query (slang + fuzzy) — CHỈ UNIGRAM
        try:
            full_vocab = set(self.vec.get_feature_names_out())
        except Exception:
            full_vocab = set()
        self.vocab = {w for w in full_vocab if (" " not in w) and (len(w) <= 30)}

        # ML ý định (tuỳ chọn)
        try:
            self.ml = joblib.load("models/ml_classifier.joblib")
        except Exception:
            self.ml = None

    # ====== Đa dạng bằng MMR (tránh trùng nội dung) ======
    def mmr_rerank(self, cand_df: pd.DataFrame, selected_ids: set[int], base_col: str = "score", lamb: float = 0.75):
        if cand_df is None or cand_df.empty or not selected_ids:
            return cand_df
        cand_idx = cand_df.index.values
        sel_mask = self.df["id"].isin(selected_ids)
        sel_idx = self.df[sel_mask].index.values
        if len(sel_idx) == 0:
            return cand_df
        S = cosine_similarity(self.X[cand_idx], self.X[sel_idx])  # [Ncand, Nsel]
        max_sim = S.max(axis=1) if S.size else np.zeros(len(cand_idx))
        rerank = cand_df.copy()
        base = rerank.get(base_col, pd.Series(np.zeros(len(rerank)), index=rerank.index)).to_numpy(dtype=float)
        rerank["mmr_penalty"] = max_sim
        rerank["score_mmr"] = lamb * base - (1 - lamb) * rerank["mmr_penalty"].values
        return rerank.sort_values("score_mmr", ascending=False)

    def normalize_for_debug(self, q: str) -> str:
        return normalize_query(q, self.vocab)

    # ========= Helper chọn text Series an toàn (tránh dùng `or` với Series) =========
    def _pick_text_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Chọn cột text dùng để match keyword/intent:
        - Ưu tiên 'text' nếu có và có ít nhất một giá trị khác NaN; fillna bằng 'name'
        - Nếu không, fallback về 'name'
        """
        if "text" in df.columns:
            ser = df["text"]
            if ser.notna().any():
                return ser.fillna(df.get("name"))
        return df["name"]

    # ====== Filter city “lỏng” ======
    def _city_filter(self, df: pd.DataFrame, city: str | None) -> pd.DataFrame:
        if not city:
            return df
        key = strip_accents(city).lower().strip()
        key = CITY_MAP.get(key, key)
        mask = df["city"].astype(str).str.lower().apply(strip_accents).str.contains(key, na=False)
        sub = df[mask]
        return sub if not sub.empty else df  # fallback

    # ====== Backoff: boost theo keyword khi cosine yếu ======
    def _kw_boost(self, df: pd.DataFrame, q_norm: str) -> pd.Series:
        """
        Backoff khi cosine yếu: cộng điểm theo match từ khóa trong tags/text.
        Trả về series [0..1] để cộng vào base (trọng số nhỏ).
        """
        if df is None or df.empty:
            return pd.Series(0.0, index=df.index if df is not None else None)

        tokens = [t for t in q_norm.split() if len(t) >= 2]
        if not tokens:
            return pd.Series(0.0, index=df.index)

        lex = set(tokens)
        # mở rộng nhẹ cho một số nhóm hay gặp
        exp_map = {
            "bien": ["bien", "beach", "sea", "ocean", "sea view", "my khe", "non nuoc"],
            "giai": ["giai tri", "amusement", "theme park", "park", "rides", "sun world", "sun wheel", "asia park"],
            "cho": ["cho dem", "market", "night market"],
            "dem": ["night", "nightlife", "night market", "sun wheel", "dragon bridge"],
            "cau": ["cau rong", "bridge", "dragon bridge"],
            "hai": ["hai san", "seafood"],
        }
        for k, vs in exp_map.items():
            if any(tok.startswith(k) for tok in tokens):
                lex.update(vs)

        # SỬA: không dùng (df.get("text") or df["name"]) nữa
        text = self._pick_text_series(df).astype(str)
        tags = (df["tags"] if "tags" in df.columns else pd.Series([""] * len(df), index=df.index)).astype(str)

        def score_row(t, tg):
            s = 0
            tl = t.lower()
            gl = tg.lower()
            for w in lex:
                if w in tl or w in gl:
                    s += 1
            return s

        raw = np.array([score_row(t, g) for t, g in zip(text, tags)], dtype=float)
        if raw.max() > 0:
            raw = raw / raw.max()
        return pd.Series(raw, index=df.index)

    # ====== Intent detection để ưu tiên chủ đề phù hợp ======
    def _detect_intent(self, q_norm: str) -> dict:
        """
        Phát hiện intent thô từ query đã chuẩn hoá để áp boost theo chủ đề.
        Trả về flags: {"fun":bool, "beach":bool, "night":bool}
        """
        q = f" {q_norm.lower()} "
        return {
            "fun": any(k in q for k in [" giai tri ", " vui choi ", " cong vien ", " park ", " theme park "]),
            "beach": any(k in q for k in [" bien ", " beach ", " my khe ", " non nuoc "]),
            "night": any(k in q for k in [" dem ", " night ", " night market ", " cho dem ", " sun wheel ", " cau rong "]),
        }

    def _intent_boost(self, df: pd.DataFrame, intent: dict) -> pd.Series:
        """
        Điểm [0..1] tăng thêm theo intent và tags/text.
        fun  → ưu tiên: theme park / rides / asia park / sun wheel / amusement / cong vien
        night→ ưu tiên: night market / sun wheel / dragon bridge / river cruise
        beach→ nếu user nói 'biển'
        Ngoài ra: nếu user nói 'đi chơi' (fun) mà KHÔNG nhắc biển, phạt nhẹ các item chỉ thuần 'beach'.
        """
        if df is None or df.empty:
            return pd.Series(0.0, index=df.index if df is not None else None)

        # SỬA: không dùng (df.get("text") or df["name"]) nữa
        text = self._pick_text_series(df).astype(str).str.lower()
        tags = (df["tags"] if "tags" in df.columns else pd.Series([""] * len(df), index=df.index)).astype(str).str.lower()

        def has_any(s, keys):
            return any(k in s for k in keys)

        fun_keys = ["amusement", "theme park", "rides", "roller", "cong vien", "khu vui choi", "asia park", "sun world", "sun wheel", "vong quay"]
        night_keys = ["night market", "cho dem", "nightlife", "sun wheel", "dragon bridge", "cau rong", "river cruise"]
        beach_keys = ["beach", "bien", "my khe", "non nuoc"]

        scores = []
        for t, g in zip(text, tags):
            s = 0.0
            src = t + " " + g
            if intent.get("fun"):
                if has_any(src, fun_keys):
                    s += 1.0
            if intent.get("night"):
                if has_any(src, night_keys):
                    s += 0.8
            if intent.get("beach"):
                if has_any(src, beach_keys):
                    s += 0.8
            else:
                # Nếu user nói "đi chơi" (fun) mà KHÔNG nhắc biển → phạt nhẹ item chỉ thuần beach
                if intent.get("fun") and (has_any(src, beach_keys) and not has_any(src, fun_keys + night_keys)):
                    s -= 0.4
            scores.append(s)

        s = np.array(scores, dtype=float)
        # Chuẩn hoá về [0,1]
        if s.max() > 0:
            s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        else:
            s = np.zeros_like(s)
        return pd.Series(s, index=df.index)

    # ====== Hàm chính ======
    def itinerary(self, query_text: str, days: int, budget_total: int, city: str | None = None):
        df = self._city_filter(self.df.copy(), city)

        # ==== Chuẩn hoá query + cosine ====
        q_norm = normalize_query(query_text, self.vocab)
        q_vec = self.vec.transform([q_norm])
        cos_all = cosine_similarity(q_vec, self.X).ravel()
        df = df.assign(cos=cos_all[df.index])

        # ==== Backoff + Intent detection ====
        intent = self._detect_intent(q_norm)
        df["kw_boost"] = self._kw_boost(df, q_norm) if q_vec.nnz == 0 else 0.0
        df["intent_boost"] = self._intent_boost(df, intent)

        # ==== Trọng số theo ML (nếu có) ====
        weights = {"sightseeing": 0.5, "food": 0.25, "hotel": 0.25}
        if self.ml is not None:
            try:
                pred = self.ml.predict([query_text])[0]
                if pred == "sightseeing":
                    weights = {"sightseeing": 0.6, "food": 0.25, "hotel": 0.15}
                elif pred == "food":
                    weights = {"sightseeing": 0.35, "food": 0.45, "hotel": 0.20}
                elif pred == "hotel":
                    weights = {"sightseeing": 0.35, "food": 0.25, "hotel": 0.40}
            except Exception:
                pass

        # ==== Cân lại trọng số: tăng liên quan khi query mơ hồ ====
        ambiguous = (q_vec.nnz <= 5) or intent.get("fun") or intent.get("night")
        w_cos = 0.70
        w_rating = 0.18
        w_pop = 0.05
        w_kw = 0.15 if ambiguous else 0.05
        w_intent = 0.15 if ambiguous else 0.05

        def base(x):
            return (
                w_cos * x["cos"] +
                w_rating * (x.get("rating", 0) / 5) +
                w_pop * (x.get("popularity", 0)) +
                w_kw * x.get("kw_boost", 0) +
                w_intent * x.get("intent_boost", 0)
            )

        S = df[df["category"] == "sightseeing"].copy()
        F = df[df["category"] == "food"].copy()
        H = df[df["category"] == "hotel"].copy()
        E = df[df["category"] == "event"].copy()

        for t in (S, F, H, E):
            if not t.empty:
                t["base"] = t.apply(base, axis=1)

        # Áp trọng số ML lên từng loại
        if not S.empty: S["base"] *= weights["sightseeing"]
        if not F.empty: F["base"] *= weights["food"]
        if not H.empty: H["base"] *= weights["hotel"]
        if not E.empty: E["base"] *= 0.9  # event nhẹ hơn chút

        per_day_budget = max(0, budget_total) // max(1, days)
        used = set()

        # Tâm toạ độ hợp lệ (bỏ (0,0)), fallback trung tâm Đà Nẵng
        df_valid = df[(df["lat"].notna()) & (df["lon"].notna()) & (df["lat"] != 0.0) & (df["lon"] != 0.0)]
        if not df_valid.empty:
            center_lat, center_lon = df_valid["lat"].median(), df_valid["lon"].median()
        else:
            center_lat, center_lon = 16.0471, 108.2068

        # ===== RNG seed theo query để "khác query khác kết quả" =====
        seed = (abs(hash(q_norm)) % (2**32 - 1)) or 42
        rng = np.random.default_rng(seed)

        def pick_topk_weighted(df_sorted: pd.DataFrame, k=3, score_col="score"):
            if df_sorted is None or df_sorted.empty:
                return None
            sub = df_sorted.head(max(1, k))
            sc = sub[score_col].to_numpy(dtype=float)
            sc = np.maximum(sc - sc.min() + 1e-6, 1e-6)
            probs = sc / sc.sum()
            idx = rng.choice(len(sub), p=probs)
            return sub.iloc[idx].to_dict()

        plan = []
        for d in range(days):
            wd = d % 7

            # ===== Hotel =====
            target_hotel = per_day_budget * 0.55
            h = (
                H[~H["id"].isin(used)]
                .copy()
                .assign(
                    dist=lambda x: [safe_dist(center_lat, center_lon, la, lo) for la, lo in zip(x["lat"], x["lon"])],
                    price_aff=lambda x: 1 / (1 + (abs(x["avg_cost"] - target_hotel) / max(1, target_hotel))),
                )
            )
            if not h.empty:
                h["score_h"] = 0.7 * h["base"] + 0.2 * h["price_aff"] + 0.1 * (1 / (1 + h["dist"]))
                h = h.sort_values(by="score_h", ascending=False)
            hotel = h.iloc[0].to_dict() if not h.empty else None
            if hotel:
                used.add(hotel["id"])

            # ===== Morning (09:00) =====
            anchor_lat = hotel["lat"] if hotel else center_lat
            anchor_lon = hotel["lon"] if hotel else center_lon
            morning_time = 9 * 60
            s1_all = (
                S[~S["id"].isin(used)]
                .copy()
                .assign(dist=lambda x: [safe_dist(anchor_lat, anchor_lon, la, lo) for la, lo in zip(x["lat"], x["lon"])])
                .assign(score=lambda x: 0.65 * x["base"] + 0.35 * (1 / (1 + x["dist"])))
                .sort_values(by="score", ascending=False)
            )
            s1_all = self.mmr_rerank(s1_all, used, base_col="score", lamb=0.75)
            s1 = s1_all[s1_all.apply(lambda r: within_open(r, morning_time, wd), axis=1)]
            if s1.empty:
                s1 = s1_all
            morning = pick_topk_weighted(s1, k=3, score_col="score_mmr" if "score_mmr" in s1.columns else "score")
            if morning:
                used.add(morning["id"])

            # ===== Lunch (11:30) =====
            lunch_time = 11 * 60 + 30
            anchor_lat = (morning or {}).get("lat", anchor_lat)
            anchor_lon = (morning or {}).get("lon", anchor_lon)
            f1_all = (
                F[~F["id"].isin(used)]
                .copy()
                .assign(dist=lambda x: [safe_dist(anchor_lat, anchor_lon, la, lo) for la, lo in zip(x["lat"], x["lon"])])
                .assign(score=lambda x: 0.7 * x["base"] + 0.3 * (1 / (1 + x["dist"])))
                .sort_values(by="score", ascending=False)
            )
            f1_all = self.mmr_rerank(f1_all, used, base_col="score", lamb=0.75)
            f1 = f1_all[f1_all.apply(lambda r: within_open(r, lunch_time, wd), axis=1)]
            if f1.empty:
                f1 = f1_all
            lunch = pick_topk_weighted(f1, k=3, score_col="score_mmr" if "score_mmr" in f1.columns else "score")
            if lunch:
                used.add(lunch["id"])

            # ===== Afternoon (15:00) =====
            afternoon_time = 15 * 60
            min_close = 16 * 60 + 30
            anchor_lat = (lunch or {}).get("lat", anchor_lat)
            anchor_lon = (lunch or {}).get("lon", anchor_lon)
            s2_all = (
                S[~S["id"].isin(used)]
                .copy()
                .assign(dist=lambda x: [safe_dist(anchor_lat, anchor_lon, la, lo) for la, lo in zip(x["lat"], x["lon"])])
                .assign(score=lambda x: 0.65 * x["base"] + 0.35 * (1 / (1 + x["dist"])))
                .sort_values(by="score", ascending=False)
            )

            def ok_afternoon(r):
                if not within_open(r, afternoon_time, wd):
                    return False
                cm = r.get("close_min", None)
                try:
                    return True if pd.isna(cm) else (cm >= min_close)
                except Exception:
                    return True

            s2_all = self.mmr_rerank(s2_all, used, base_col="score", lamb=0.75)
            s2 = s2_all[s2_all.apply(ok_afternoon, axis=1)]
            if s2.empty:
                s2 = s2_all
            afternoon = pick_topk_weighted(s2, k=3, score_col="score_mmr" if "score_mmr" in s2.columns else "score")
            if afternoon:
                used.add(afternoon["id"])

            # ===== Evening Event (19:00–21:30) =====
            evening_time = 19 * 60
            min_close_e = 21 * 60 + 30
            anchor_lat_e = (afternoon or {}).get("lat", None)
            anchor_lon_e = (afternoon or {}).get("lon", None)
            if anchor_lat_e is None or anchor_lon_e is None:
                anchor_lat_e = (hotel or {}).get("lat", center_lat)
                anchor_lon_e = (hotel or {}).get("lon", center_lon)

            if not E.empty:
                e_all = (
                    E[~E["id"].isin(used)]
                    .copy()
                    .assign(dist=lambda x: [safe_dist(anchor_lat_e, anchor_lon_e, la, lo) for la, lo in zip(x["lat"], x["lon"])])
                )

                q_lc = q_norm.lower()

                def season_match_score(bt: str) -> float:
                    if not isinstance(bt, str) or not bt:
                        return 0.0
                    bt_l = bt.lower()
                    keys = ["spring", "summer", "autumn", "fall", "winter", "all year"]
                    return 0.2 if any(k in q_lc and k in bt_l for k in keys) else (0.1 if "all year" in bt_l else 0.0)

                e_all = e_all.assign(
                    seas=lambda x: x["best_time"].fillna("").apply(season_match_score),
                    score=lambda x: 0.65 * x["base"] + 0.25 * (1 / (1 + x["dist"])) + 0.10 * x["seas"],
                ).sort_values(by="score", ascending=False)

                e_all = self.mmr_rerank(e_all, used, base_col="score", lamb=0.75)

                def ok_evening(r):
                    if not within_open(r, evening_time, wd):
                        return False
                    cm = r.get("close_min", None)
                    try:
                        return True if pd.isna(cm) else (cm >= min_close_e)
                    except Exception:
                        return True

                e_ok = e_all[e_all.apply(ok_evening, axis=1)]
                if e_ok.empty:
                    e_ok = e_all

                evening = pick_topk_weighted(e_ok, k=3, score_col="score_mmr" if "score_mmr" in e_ok.columns else "score")
            else:
                evening = None

            if evening:
                used.add(evening["id"])

            # ===== Tổng chi phí ngày & đóng gói =====
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
