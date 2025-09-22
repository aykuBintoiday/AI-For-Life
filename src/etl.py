# src/etl.py
import re, unicodedata
import pandas as pd

SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
DAY_MAP = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)

def clean_text(s: str) -> str:
    s = str(s or "").strip()
    s = URL_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s)
    return s

def hhmm_to_min(x) -> int | None:
    if not isinstance(x, str):
        return None
    x = x.strip()
    if not x:
        return None
    try:
        h, m = x.split(":")
        return int(h) * 60 + int(m)
    except:
        return None

def price_to_level(x) -> int:
    if not isinstance(x, str):
        return 0
    x = x.strip()
    return 1 if x == "$" else 2 if x == "$$" else 3 if x == "$$$" else 0

def parse_closed_days(x) -> set[int]:
    if not isinstance(x, str):
        return set()
    x = x.strip()
    if not x:
        return set()
    parts = [p.strip() for p in re.split(r"[,\s;/]+", x) if p.strip()]
    out = set()
    for p in parts:
        if p in DAY_MAP:
            out.add(DAY_MAP[p])
    return out

# ===== Synonyms để tăng phủ từ khoá phổ biến =====
SYNONYMS = {
    "biển": ["bien","beach","sea","bai bien","ocean","sea view","view bien"],
    "chụp ảnh": ["chup anh","chup hinh","photo","photography","checkin","check-in"],
    "văn hoá": ["van hoa","culture","cultural"],
    "bảo tàng": ["bao tang","museum"],
    "chợ": ["cho","market","night market","cho dem"],
    "cà phê": ["ca phe","coffee","cafe"],
    "hải sản": ["hai san","seafood"],
    "khách sạn": ["khach san","hotel"],
    "nghỉ dưỡng": ["nghi duong","resort","spa"],
    "cầu": ["cau","bridge"],
    "đi thuyền": ["di thuyen","boat","basket boat","river cruise"],
    "ngắm cảnh": ["ngam canh","viewpoint","rooftop","sunset","sunrise","night view"],
    "đặc sản": ["dac san","local food","street food"],

    # ==== BỔ SUNG cho 'đi chơi' ====
    "giải trí": ["giai tri","amusement","park","theme park","rides","roller coaster","cong vien","khu vui choi"],
    "vui chơi": ["vui choi","amusement","park","theme park","rides","cong vien","khu vui choi"],
    "đi dạo": ["di dao","walking","promenade","park","riverfront"],
    "đi chơi": ["giai tri","vui choi","cong vien","amusement","park","night market","cho dem","bar","nightlife"],

    # ===== ĐỊA DANH ĐÀ NẴNG =====
    "ba na": ["ba na hills","bana","bà nà","banahills","sun world ba na","sunworld bana"],
    "bà nà": ["ba na hills","bana","ba na","banahills","sun world ba na","sunworld bana"],
    "sun world": ["sun world ba na","sunworld bana","ba na hills","danang wonders","asia park"],
    "cau rong": ["cầu rồng","dragon bridge","cau rong da nang","dragonbridge"],
    "cầu rồng": ["cau rong","dragon bridge","cau rong da nang","dragonbridge"],
    "son tra": ["bán đảo sơn trà","son tra peninsula","linh ung","chua linh ung","bai but"],
    "sơn trà": ["ban dao son tra","son tra peninsula","linh ung","chua linh ung","bai but"],
    "my khe": ["bãi biển mỹ khê","my khe beach"],
    "núi thần tài": ["suoi nuoc nong","than tai mountain hot spring","nuoc nong","hot spring park"],
    "hoi an": ["hội an","ancient town","pho co hoi an"],
}

def expand_with_synonyms(text_lower_nodiac: str) -> str:
    extra = []
    for k, vs in SYNONYMS.items():
        k_norm = strip_accents(k).lower()
        if k_norm in text_lower_nodiac:
            extra.extend(vs)
    return " ".join(sorted(set(extra)))

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Chuẩn hoá cột string
    df["name"] = df["name"].astype(str)
    df["city"] = df["city"].astype(str)
    df["category"] = df["category"].astype(str).str.lower()
    df["desc"] = df["desc"].apply(clean_text)
    df["tags"] = df["tags"].fillna("").astype(str)
    df["best_time"] = df["best_time"].fillna("").astype(str)
    df["address"] = df["address"].fillna("").astype(str)

    # Ép chuỗi cho các cột có thể NaN
    for c in ["open_time", "close_time", "closed_days", "price_level"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    # Chuẩn hoá số
    for c in ["duration_min","avg_cost","rating","review_count","popularity","lat","lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Giờ mở/đóng + ngày nghỉ
    df["open_min"]  = df["open_time"].apply(hhmm_to_min)
    df["close_min"] = df["close_time"].apply(hhmm_to_min)
    df["closed_set"] = df["closed_days"].apply(parse_closed_days)

    # Price level ($, $$, $$$) → 1..3
    df["price_level_num"] = df["price_level"].apply(price_to_level)

    # Tên không dấu để tra cứu
    df["norm_name"] = df["name"].apply(lambda s: strip_accents(s).lower())

    # Văn bản hợp nhất cho TF-IDF + synonyms
    base_text = (
        df["name"].astype(str) + " " + df["city"].astype(str) + " " +
        df["category"].astype(str) + " " + df["tags"].astype(str) + " " +
        df["desc"].astype(str)
    ).apply(lambda s: strip_accents(clean_text(s)).lower())

    df["text"] = base_text + " " + base_text.apply(expand_with_synonyms)

    # Điền thiếu an toàn
    df["duration_min"] = df["duration_min"].fillna(90).astype(int)
    df["avg_cost"] = df["avg_cost"].fillna(0).astype(int)
    df["rating"] = df["rating"].fillna(0.0)
    df["lat"] = df["lat"].fillna(0.0)
    df["lon"] = df["lon"].fillna(0.0)

    return df


# ===== CLI nhỏ để ghi parquet sạch phục vụ build_index =====
if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "data/itinerary_dataset.csv"
    df = load_dataset(src).reset_index(drop=True)
    outp = "data/itinerary_dataset.clean.parquet"
    df.to_parquet(outp, index=False)
    print(f"✅ Wrote {outp} with {len(df)} rows")
