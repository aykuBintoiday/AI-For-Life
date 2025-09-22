# train_ml.py
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import os

DATA = "training_queries.csv"
OUT  = "models/ml_classifier.joblib"

def main():
    df = pd.read_csv(DATA)
    df["query"] = df["query"].astype(str)
    df["label"] = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        df["query"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("ACC:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, OUT)
    print(f"✅ Saved {OUT}")

if __name__ == "__main__":
    main()
