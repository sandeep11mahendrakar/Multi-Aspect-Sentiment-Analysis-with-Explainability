"""Reproducible classical-baseline training pipeline.

Replicates the exact pipeline of notebooks/01_EDA.ipynb + 02_Baseline_Models.ipynb:
  rating >= 4 -> positive, rating == 3 -> neutral, rating <= 2 -> negative
  clean -> stopword removal (keeping negations) -> lemmatize
  stratified 80/20 split, random_state=42
  TF-IDF(max_features=5000, ngram_range=(1,2), min_df=2)
  LogisticRegression(max_iter=1000, random_state=42)

Run from repo root:
    venv\\Scripts\\python.exe src\\train.py
"""

import json
import os
import re
import sys

import joblib
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(REPO_ROOT, "data", "raw", "reviews.csv")
CLEANED_CSV = os.path.join(REPO_ROOT, "data", "processed", "cleaned_reviews.csv")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
METRICS_DIR = os.path.join(REPO_ROOT, "results", "metrics")

RANDOM_STATE = 42


def ensure_nltk():
    for pkg in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)


def rating_to_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating <= 2:
        return "negative"
    return "neutral"


NEGATIONS = {"not", "no", "nor", "never"}


def build_pipeline_parts():
    """Return (basic_clean, preprocess) replicating notebook 01 exactly."""
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stop_words = set(stopwords.words("english")) - NEGATIONS
    lemmatizer = WordNetLemmatizer()

    def basic_clean(text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def preprocess(text):
        tokens = basic_clean(text).split()
        tokens = [w for w in tokens if w not in stop_words]
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        return " ".join(tokens)

    return basic_clean, preprocess


def main():
    ensure_nltk()
    _, preprocess = build_pipeline_parts()

    print(f"[1/6] Loading raw data: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    print(f"      raw shape: {df.shape}")

    print("[2/6] Cleaning text (negation-preserving stopwords + lemmatization)...")
    df["sentiment"] = df["Rating"].apply(rating_to_sentiment)
    df["final_text"] = df["Review Text"].fillna("").apply(preprocess)

    # NOTE: empty reviews are written as empty CSV fields; reloading converts
    # them to NaN and notebook 02's dropna removes them (this reproduces the
    # original n=4,529 test set).
    os.makedirs(os.path.dirname(CLEANED_CSV), exist_ok=True)
    df[["final_text", "sentiment"]].to_csv(CLEANED_CSV, index=False)
    df = pd.read_csv(CLEANED_CSV).dropna(subset=["final_text", "sentiment"])
    print(f"      usable rows after dropna: {len(df)}")

    dist = df["sentiment"].value_counts()
    dist_pct = df["sentiment"].value_counts(normalize=True).round(4)
    print("[3/6] Class distribution:")
    for cls in ["positive", "neutral", "negative"]:
        print(f"      {cls:<10} {dist.get(cls, 0):>6}  ({dist_pct.get(cls, 0)*100:.1f}%)")

    X = df["final_text"]
    y = df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[4/6] Split: train={len(X_train)}, test={len(X_test)} (seed={RANDOM_STATE}, stratified)")

    print("[5/6] Vectorizing (TF-IDF 5000 features, 1-2 grams, min_df=2)...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("[6/6] Training LogisticRegression(max_iter=1000, seed=42)...")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "baseline_logreg.pkl"))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

    os.makedirs(METRICS_DIR, exist_ok=True)
    preds_path = os.path.join(METRICS_DIR, "baseline_predictions.csv")
    pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred}).to_csv(preds_path, index=False)

    meta = {
        "random_state": RANDOM_STATE,
        "n_rows": int(len(df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "class_distribution": {k: int(v) for k, v in dist.items()},
        "vectorizer": {"max_features": 5000, "ngram_range": [1, 2], "min_df": 2},
        "model": "LogisticRegression(max_iter=1000, random_state=42)",
        "label_mapping": "rating>=4 positive | rating==3 neutral | rating<=2 negative",
    }
    with open(os.path.join(MODELS_DIR, "train_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nTraining complete.")
    print(f"  model       -> models/baseline_logreg.pkl")
    print(f"  vectorizer  -> models/tfidf_vectorizer.pkl")
    print(f"  predictions -> {os.path.relpath(preds_path, REPO_ROOT)}")
    print(f"Next: python src\\evaluate.py")


if __name__ == "__main__":
    sys.exit(main())
