# src/preprocessing.py
# ─────────────────────────────────────────────────────────────────────────────
# Builds all feature matrices from the already-split train/val/test CSVs.
# Run AFTER create_splits.py.  Saves vectorizers + sparse feature matrices
# to data/processed/ and models/model_a/traditional/.
#
# Hard rules enforced here:
#   - fit_transform() on TRAIN only;  transform() on val/test
#   - Keep matrices sparse (never call .toarray() on full dataset)
#   - class_weight='balanced' is NOT applied here; applied during training
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs("data/processed",                    exist_ok=True)
os.makedirs("models/model_a/traditional",        exist_ok=True)
os.makedirs("models/model_b/traditional",        exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    print(f"  ✓ Checkpoint saved : {path}")


def load_checkpoint(path: str):
    if os.path.exists(path):
        print(f"  ↩ Loading checkpoint : {path}")
        return joblib.load(path)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Text utilities
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> list:
    """Split text into sentences of at least 10 characters."""
    parts = re.split(r"(?<=[.!?])\s+", str(text))
    return [s.strip() for s in parts if len(s.strip()) > 10]


# ══════════════════════════════════════════════════════════════════════════════
# Row expansion  (each MCQ row → 4 binary classification samples)
# ══════════════════════════════════════════════════════════════════════════════

def expand_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        combined     : "{article} {article} {question} {option}"  (article doubled for weight)
        article      : cleaned article text
        question     : cleaned question text
        option       : cleaned option text
        option_label : A / B / C / D
        label        : 1 if correct, 0 otherwise

    Expected class distribution after expansion: ~25 % positive, ~75 % negative.
    """
    rows = []
    for _, row in df.iterrows():
        article  = clean_text(row["article"])
        question = clean_text(row["question"])
        for opt_label in ["A", "B", "C", "D"]:
            option_text = clean_text(str(row[opt_label]))
            label       = 1 if row["answer"] == opt_label else 0
            # Article repeated twice to give it extra weight in OHE/TF-IDF
            combined    = f"{article} {article} {question} {option_text}"
            rows.append({
                "combined":     combined,
                "article":      article,
                "question":     question,
                "option":       option_text,
                "option_label": opt_label,
                "label":        label,
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Cosine similarity features  (6 scalars per sample)
# ══════════════════════════════════════════════════════════════════════════════

def compute_cosine_features(
    vectorizer,
    articles: pd.Series,
    questions: pd.Series,
    options: pd.Series,
    batch_size: int = 1000,
) -> np.ndarray:
    """
    Returns shape (n_samples, 3) with:
        col 0 : cosine_similarity(article, question)
        col 1 : cosine_similarity(article, option)
        col 2 : cosine_similarity(question, option)

    Processed in batches to avoid memory spikes.
    """
    arts  = articles.tolist()
    qs    = questions.tolist()
    opts  = options.tolist()
    feats = []

    for i in range(0, len(arts), batch_size):
        art_vecs  = vectorizer.transform(arts[i : i + batch_size])
        q_vecs    = vectorizer.transform(qs[i   : i + batch_size])
        opt_vecs  = vectorizer.transform(opts[i  : i + batch_size])

        # element-wise dot product on sparse rows, then normalise
        def row_cosine(A, B):
            # returns 1-D array of per-row cosines between two sparse matrices
            norms_A = np.sqrt(A.multiply(A).sum(axis=1)).A1
            norms_B = np.sqrt(B.multiply(B).sum(axis=1)).A1
            dots    = A.multiply(B).sum(axis=1).A1
            denom   = norms_A * norms_B
            denom[denom == 0] = 1e-9
            return dots / denom

        batch_feats = np.column_stack([
            row_cosine(art_vecs, q_vecs),
            row_cosine(art_vecs, opt_vecs),
            row_cosine(q_vecs,   opt_vecs),
        ])
        feats.append(batch_feats)

    return np.vstack(feats)


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_preprocessing(max_train_rows: int = None):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    max_train_rows : int or None
        If set, use only the first N rows of train for faster iteration.
        Set to None for full dataset.  Recommend starting with 30_000 on Colab.
    """

    # ── Load splits ──────────────────────────────────────────────────────────
    print("Loading splits …")
    train_df = pd.read_csv("data/raw/train.csv")
    val_df   = pd.read_csv("data/raw/val.csv")
    test_df  = pd.read_csv("data/raw/test.csv")

    if max_train_rows:
        print(f"  [DEBUG] Limiting train to {max_train_rows} rows.")
        train_df = train_df.head(max_train_rows)

    print(f"  Train {len(train_df):,} | Val {len(val_df):,} | Test {len(test_df):,}")

    # ── Expand to binary ─────────────────────────────────────────────────────
    print("\nExpanding rows to binary classification samples …")
    train_exp = expand_to_binary(train_df)
    val_exp   = expand_to_binary(val_df)
    test_exp  = expand_to_binary(test_df)

    print(f"  Train expanded : {len(train_exp):,}  (label mean={train_exp['label'].mean():.3f})")
    print(f"  Val expanded   : {len(val_exp):,}")
    print(f"  Test expanded  : {len(test_exp):,}")

    # Save expanded DataFrames (needed by model_b_train.py)
    train_exp.to_csv("data/processed/train_expanded.csv", index=False)
    val_exp.to_csv(  "data/processed/val_expanded.csv",   index=False)
    test_exp.to_csv( "data/processed/test_expanded.csv",  index=False)
    print("  Expanded CSVs saved to data/processed/")

    y_train = train_exp["label"]
    y_val   = val_exp["label"]
    y_test  = test_exp["label"]

    # ── OHE vectorizer (primary, required) ───────────────────────────────────
    print("\nFitting OHE (CountVectorizer binary=True) …")
    ohe_path = "models/model_a/traditional/ohe_vectorizer.pkl"
    ohe_vec  = load_checkpoint(ohe_path)

    if ohe_vec is None:
        ohe_vec = CountVectorizer(
            max_features=10_000,
            stop_words="english",
            binary=True,
            min_df=2,
            max_df=0.95,
        )
        ohe_vec.fit(train_exp["combined"])
        save_checkpoint(ohe_vec, ohe_path)

    X_train_ohe = ohe_vec.transform(train_exp["combined"])
    X_val_ohe   = ohe_vec.transform(val_exp["combined"])
    X_test_ohe  = ohe_vec.transform(test_exp["combined"])
    print(f"  OHE shape : {X_train_ohe.shape}")

    # ── TF-IDF vectorizer (optional, recommended) ────────────────────────────
    print("\nFitting TF-IDF vectorizer …")
    tfidf_path = "models/model_a/traditional/tfidf_vectorizer.pkl"
    tfidf_vec  = load_checkpoint(tfidf_path)

    if tfidf_vec is None:
        tfidf_vec = TfidfVectorizer(
            max_features=10_000,
            stop_words="english",
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )
        tfidf_vec.fit(train_exp["combined"])
        save_checkpoint(tfidf_vec, tfidf_path)

    X_train_tfidf = tfidf_vec.transform(train_exp["combined"])
    X_val_tfidf   = tfidf_vec.transform(val_exp["combined"])
    X_test_tfidf  = tfidf_vec.transform(test_exp["combined"])
    print(f"  TF-IDF shape : {X_train_tfidf.shape}")

    # ── Cosine similarity features ────────────────────────────────────────────
    print("\nComputing cosine similarity features …")

    cos_train = compute_cosine_features(
        ohe_vec, train_exp["article"], train_exp["question"], train_exp["option"]
    )
    cos_val   = compute_cosine_features(
        ohe_vec, val_exp["article"],   val_exp["question"],   val_exp["option"]
    )
    cos_test  = compute_cosine_features(
        ohe_vec, test_exp["article"],  test_exp["question"],  test_exp["option"]
    )
    print(f"  Cosine feature shape : {cos_train.shape}")

    # ── Combine OHE + cosine features (primary feature matrix) ───────────────
    print("\nCombining OHE + cosine features into X_*_full …")
    X_train_full = sp.hstack([X_train_ohe, sp.csr_matrix(cos_train)])
    X_val_full   = sp.hstack([X_val_ohe,   sp.csr_matrix(cos_val)])
    X_test_full  = sp.hstack([X_test_ohe,  sp.csr_matrix(cos_test)])
    print(f"  X_train_full shape : {X_train_full.shape}")

    # ── Save all matrices ─────────────────────────────────────────────────────
    print("\nSaving sparse matrices …")
    sp.save_npz("data/processed/X_train_ohe.npz",   X_train_ohe)
    sp.save_npz("data/processed/X_val_ohe.npz",     X_val_ohe)
    sp.save_npz("data/processed/X_test_ohe.npz",    X_test_ohe)

    sp.save_npz("data/processed/X_train_tfidf.npz", X_train_tfidf)
    sp.save_npz("data/processed/X_val_tfidf.npz",   X_val_tfidf)
    sp.save_npz("data/processed/X_test_tfidf.npz",  X_test_tfidf)

    sp.save_npz("data/processed/X_train_full.npz",  X_train_full)
    sp.save_npz("data/processed/X_val_full.npz",    X_val_full)
    sp.save_npz("data/processed/X_test_full.npz",   X_test_full)

    # Save labels
    np.save("data/processed/y_train.npy", y_train.values)
    np.save("data/processed/y_val.npy",   y_val.values)
    np.save("data/processed/y_test.npy",  y_test.values)

    print("\n Preprocessing complete.  All artifacts in data/processed/ and models/")
    return {
        "ohe_vec":       ohe_vec,
        "tfidf_vec":     tfidf_vec,
        "X_train_full":  X_train_full,
        "X_val_full":    X_val_full,
        "X_test_full":   X_test_full,
        "X_train_ohe":   X_train_ohe,
        "X_val_ohe":     X_val_ohe,
        "X_test_ohe":    X_test_ohe,
        "y_train":       y_train.values,
        "y_val":         y_val.values,
        "y_test":        y_test.values,
        "train_exp":     train_exp,
        "val_exp":       val_exp,
        "test_exp":      test_exp,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Change max_train_rows=None to use the full dataset.
    # Start with 30_000 on free Colab to validate the pipeline first.
    run_preprocessing(max_train_rows=None)
