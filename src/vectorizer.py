# src/vectorizer.py
import os, joblib, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

VEC_PATH = "models/tfidf_vectorizer.joblib"
MAT_PATH = "models/tfidf_matrix.npz"

def build_tfidf(texts: list[str]):
    vec = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=1,
        max_df=0.98,
        strip_accents=None,      # đã strip ở ETL
        lowercase=False,         # đã lowercase ở ETL
        analyzer="word"
    )
    X = vec.fit_transform(texts)
    return vec, X

def save_tfidf(vec, X, vec_path=VEC_PATH, mat_path=MAT_PATH):
    os.makedirs(os.path.dirname(vec_path), exist_ok=True)
    joblib.dump(vec, vec_path)
    sp.save_npz(mat_path, X)

def load_tfidf(vec_path=VEC_PATH, mat_path=MAT_PATH):
    vec = joblib.load(vec_path)
    X = sp.load_npz(mat_path)
    return vec, X
