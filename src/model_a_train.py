# src/model_a_train.py
# ── COLAB SETUP (run these in a Colab cell before executing this script) ──────
# !git clone https://github.com/YOUR_USERNAME/race_rc_project.git /content/race_rc_project
# import os; os.chdir('/content/race_rc_project')
# !pip install -r requirements.txt -q
# ── Then run: python src/model_a_train.py ─────────────────────────────────────

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import LinearSVC
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.naive_bayes     import MultinomialNB
from sklearn.ensemble        import VotingClassifier
from sklearn.cluster         import KMeans
from sklearn.mixture         import GaussianMixture
from sklearn.semi_supervised import LabelPropagation
from sklearn.decomposition   import TruncatedSVD
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     f1_score, accuracy_score, silhouette_score)
from scipy.stats import mode

from src.preprocessing import save_checkpoint, load_checkpoint

MODEL_DIR = os.path.join("models", "model_a", "traditional")
os.makedirs(MODEL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Data loader
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("Loading preprocessed matrices …")
    X_train     = sp.load_npz(os.path.join("data", "processed", "X_train_full.npz"))
    X_val       = sp.load_npz(os.path.join("data", "processed", "X_val_full.npz"))
    X_test      = sp.load_npz(os.path.join("data", "processed", "X_test_full.npz"))
    X_train_ohe = sp.load_npz(os.path.join("data", "processed", "X_train_ohe.npz"))
    X_val_ohe   = sp.load_npz(os.path.join("data", "processed", "X_val_ohe.npz"))
    y_train     = np.load(os.path.join("data", "processed", "y_train.npy"))
    y_val       = np.load(os.path.join("data", "processed", "y_val.npy"))
    y_test      = np.load(os.path.join("data", "processed", "y_test.npy"))

    print(f"  X_train_full : {X_train.shape}  label mean={y_train.mean():.3f}")
    print(f"  X_val_full   : {X_val.shape}")
    return X_train, X_val, X_test, X_train_ohe, X_val_ohe, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helper
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(name, model, X_v, y_v):
    y_pred = model.predict(X_v)
    acc    = accuracy_score(y_v, y_pred)
    mf1    = f1_score(y_v, y_pred, average="macro")
    cm     = confusion_matrix(y_v, y_pred)
    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    print(f"  Accuracy : {acc:.4f}  |  Macro F1 : {mf1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(classification_report(y_v, y_pred, target_names=["incorrect", "correct"]))
    return {"accuracy": float(acc), "macro_f1": float(mf1), "confusion_matrix": cm.tolist()}


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Supervised
# ══════════════════════════════════════════════════════════════════════════════

def train_logistic_regression(X_train, y_train, X_val, y_val):
    path  = os.path.join(MODEL_DIR, "lr_model.pkl")
    model = load_checkpoint(path)
    if model is None:
        print("\nTraining Logistic Regression …")
        model = LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=1000, solver="lbfgs", random_state=42)
        model.fit(X_train, y_train)
        save_checkpoint(model, path)
    evaluate_model("Logistic Regression (val)", model, X_val, y_val)
    return model


def train_svm(X_train, y_train, X_val, y_val):
    path  = os.path.join(MODEL_DIR, "svm_model.pkl")
    model = load_checkpoint(path)
    if model is None:
        print("\nTraining LinearSVC (calibrated) …")
        base  = LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
        model = CalibratedClassifierCV(base, cv=3)
        model.fit(X_train, y_train)
        save_checkpoint(model, path)
    evaluate_model("SVM / LinearSVC (val)", model, X_val, y_val)
    return model


def train_naive_bayes(X_train_ohe, y_train, X_val_ohe, y_val):
    """Uses OHE matrix only — NB requires non-negative features."""
    path  = os.path.join(MODEL_DIR, "nb_model.pkl")
    model = load_checkpoint(path)
    if model is None:
        print("\nTraining Naive Bayes …")
        model = MultinomialNB(alpha=1.0)
        model.fit(X_train_ohe, y_train)
        save_checkpoint(model, path)
    evaluate_model("Naive Bayes (val)", model, X_val_ohe, y_val)
    return model


def train_random_forest(X_train, y_train, X_val, y_val, max_rows=50_000):
    """RF trained on first 50 000 dense rows — documented limitation."""
    from sklearn.ensemble import RandomForestClassifier
    path  = os.path.join(MODEL_DIR, "rf_model.pkl")
    model = load_checkpoint(path)
    if model is None:
        print(f"\nTraining Random Forest on first {max_rows:,} rows (dense subset) …")
        X_dense = X_train[:max_rows].toarray()
        model   = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                         random_state=42, n_jobs=-1)
        model.fit(X_dense, y_train[:max_rows])
        save_checkpoint(model, path)
    X_val_dense = X_val[:max_rows].toarray()
    evaluate_model("Random Forest (val, dense subset)", model, X_val_dense, y_val[:max_rows])
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Unsupervised
# ══════════════════════════════════════════════════════════════════════════════

def _cluster_purity(true_labels, cluster_labels):
    total = 0
    for cid in np.unique(cluster_labels):
        mask     = cluster_labels == cid
        majority = mode(true_labels[mask], keepdims=True).mode[0]
        total   += np.sum(true_labels[mask] == majority)
    return total / len(true_labels)


def train_unsupervised(X_train_ohe, X_val_ohe, y_train, y_val):
    # ── SVD ──────────────────────────────────────────────────────────────────
    svd_path = os.path.join(MODEL_DIR, "svd_reducer.pkl")
    svd      = load_checkpoint(svd_path)
    if svd is None:
        print("\nFitting TruncatedSVD (100 components) …")
        svd = TruncatedSVD(n_components=100, random_state=42)
        svd.fit(X_train_ohe)
        save_checkpoint(svd, svd_path)

    X_train_r = svd.transform(X_train_ohe)
    X_val_r   = svd.transform(X_val_ohe)
    print(f"  Reduced shape : {X_train_r.shape}")
    results = {}

    # ── KMeans ───────────────────────────────────────────────────────────────
    km_path = os.path.join(MODEL_DIR, "kmeans_model.pkl")
    km      = load_checkpoint(km_path)
    if km is None:
        print("\nTraining KMeans (k=2) …")
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        km.fit(X_train_r)
        save_checkpoint(km, km_path)

    km_preds = km.predict(X_val_r)
    sil_km   = silhouette_score(X_val_r, km_preds, sample_size=5000, random_state=42)
    purity   = _cluster_purity(y_val, km_preds)
    print(f"KMeans  Silhouette={sil_km:.4f}  Purity={purity:.4f}")
    results["kmeans_silhouette"] = float(sil_km)
    results["kmeans_purity"]     = float(purity)

    # ── GaussianMixture ───────────────────────────────────────────────────────
    gmm_path = os.path.join(MODEL_DIR, "gmm_model.pkl")
    gmm      = load_checkpoint(gmm_path)
    if gmm is None:
        print("\nTraining GaussianMixture (k=2) …")
        gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
        gmm.fit(X_train_r)
        save_checkpoint(gmm, gmm_path)

    gmm_preds = gmm.predict(X_val_r)
    sil_gmm   = silhouette_score(X_val_r, gmm_preds, sample_size=5000, random_state=42)
    purity_g  = _cluster_purity(y_val, gmm_preds)
    print(f"GMM     Silhouette={sil_gmm:.4f}  Purity={purity_g:.4f}")

    # ── LabelPropagation ─────────────────────────────────────────────────────
    lp_path  = os.path.join(MODEL_DIR, "label_prop_model.pkl")
    lp_model = load_checkpoint(lp_path)
    if lp_model is None:
        print("\nTraining LabelPropagation (10 % labelled) …")
        n_labeled = int(0.10 * len(X_train_r))
        y_semi    = y_train.copy().astype(float)
        y_semi[n_labeled:] = -1
        lp_model = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=1000)
        lp_model.fit(X_train_r, y_semi)
        save_checkpoint(lp_model, lp_path)

    lp_preds = lp_model.predict(X_val_r)
    lp_f1    = f1_score(y_val, lp_preds, average="macro")
    print(f"LabelProp  Macro F1={lp_f1:.4f}")
    results["label_prop_f1"] = float(lp_f1)

    return X_train_r, X_val_r, results


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5 — Ensemble
# ══════════════════════════════════════════════════════════════════════════════

def train_ensemble(lr_model, svm_model, X_train, y_train, X_val, y_val):
    path  = os.path.join(MODEL_DIR, "ensemble_model.pkl")
    model = load_checkpoint(path)
    if model is None:
        print("\nTraining soft-voting ensemble (LR + SVM) …")
        model = VotingClassifier(
            estimators=[("lr", lr_model), ("svm", svm_model)],
            voting="soft",
        )
        model.fit(X_train, y_train)
        save_checkpoint(model, path)
    evaluate_model("Ensemble LR+SVM (val)", model, X_val, y_val)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    X_train, X_val, X_test, X_train_ohe, X_val_ohe, y_train, y_val, y_test = load_data()

    lr_model  = train_logistic_regression(X_train, y_train, X_val, y_val)
    svm_model = train_svm(X_train, y_train, X_val, y_val)
    nb_model  = train_naive_bayes(X_train_ohe, y_train, X_val_ohe, y_val)
    rf_model  = train_random_forest(X_train, y_train, X_val, y_val)

    X_train_r, X_val_r, unsup_results = train_unsupervised(
        X_train_ohe, X_val_ohe, y_train, y_val
    )

    ensemble_model = train_ensemble(lr_model, svm_model, X_train, y_train, X_val, y_val)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  RESULTS SUMMARY (validation set)")
    print("="*55)
    for name, model, X_v, y_v in [
        ("LR",       lr_model,       X_val,     y_val),
        ("SVM",      svm_model,      X_val,     y_val),
        ("NB",       nb_model,       X_val_ohe, y_val),
        ("Ensemble", ensemble_model, X_val,     y_val),
    ]:
        f1 = f1_score(y_v, model.predict(X_v), average="macro")
        print(f"  {name:<12} Macro F1 = {f1:.4f}")

    print("\nUnsupervised:")
    for k, v in unsup_results.items():
        print(f"  {k:<25} {v:.4f}")

    summary = {
        "lr":            {"macro_f1": float(f1_score(y_val, lr_model.predict(X_val),       average="macro"))},
        "svm":           {"macro_f1": float(f1_score(y_val, svm_model.predict(X_val),      average="macro"))},
        "nb":            {"macro_f1": float(f1_score(y_val, nb_model.predict(X_val_ohe),   average="macro"))},
        "ensemble":      {"macro_f1": float(f1_score(y_val, ensemble_model.predict(X_val), average="macro"))},
        "unsupervised":  unsup_results,
    }
    os.makedirs("models", exist_ok=True)
    with open(os.path.join("models", "model_a_val_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n model_a_train.py complete.")
    print("   Results  → models/model_a_val_results.json")
    print("   Saved    → models/model_a/traditional/*.pkl")


if __name__ == "__main__":
    main()
