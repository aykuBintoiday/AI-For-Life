# src/textnorm.py
import re, unicodedata
from rapidfuzz import process, fuzz

SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")

# 1) Bỏ dấu, hạ chữ, gộp khoảng trắng, bỏ URL
def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)

def basic_clean(s: str) -> str:
    s = s or ""
    s = str(s).strip()
    s = URL_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s)
    return s

# 2) Đồng nghĩa/tiếng lóng gần gũi → từ chuẩn
VN_SLANG = {
    "sinh to song ao": "check in",
    "song ao": "check in",
    "checkin": "check in",
    "an uong": "am thuc",
    "an ngon": "am thuc",
    "song bien": "bien",
    "bien dep": "bien",
    "tam bien": "bien",
    "nghi duong": "resort",
    "tham quan": "ngam canh",
    "chup hinh": "chup anh",

    # ==== BỔ SUNG cho truy vấn phổ thông ====
    "di choi": "giai tri",
    "vui choi": "giai tri",
    "choi": "giai tri",
    "di dao": "di dao",
    "cho dem": "night market",
    "di dem": "nightlife",
    "an hai san": "hai san",
}

def apply_slang(s: str) -> str:
    s2 = s
    for k, v in VN_SLANG.items():
        s2 = re.sub(rf"\b{k}\b", v, s2)
    return s2

# 3) Sửa lỗi chính tả “vừa đủ”
def fuzzy_fix_tokens(text: str, vocab: set[str], score_cutoff: int = 85) -> str:
    tokens = text.split()
    fixed = []
    for t in tokens:
        # nếu từ đã nằm trong vocab thì giữ nguyên
        if t in vocab or t.isdigit():
            fixed.append(t); continue
        # tìm từ gần nhất trong vocab (UNIGRAM ONLY đã được lọc ở Recommender)
        cand, score, _ = process.extractOne(
            t, vocab, scorer=fuzz.WRatio, score_cutoff=score_cutoff
        ) or (None, 0, None)
        fixed.append(cand if cand else t)
    return " ".join(fixed)

def normalize_query(q: str, vocab: set[str]) -> str:
    q0 = basic_clean(q).lower()
    q1 = strip_accents(q0)           # không dấu để robust
    q2 = apply_slang(q1)             # map tiếng lóng
    q3 = fuzzy_fix_tokens(q2, vocab) # sửa lỗi chính tả nhẹ
    return q3
