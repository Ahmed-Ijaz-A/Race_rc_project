# src/evaluate.py
# ── COLAB SETUP (run these in a Colab cell before executing this script) ──────
# !git clone https://github.com/YOUR_USERNAME/race_rc_project.git /content/race_rc_project
# import os; os.chdir('/content/race_rc_project')
# !pip install -r requirements.txt -q
# ── Then run: python src/evaluate.py ──────────────────────────────────────────

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report,
                             silhouette_score)
from scipy.stats import mode
from evaluate import load as hf_load

from src.preprocessing import load_checkpoint, clean_text
from src.inference import (predict_answer, generate_distractors,
                           generate_hints, load_all_models)

RESULT_PATH = os.path.join("models", "eval_results.json")
os.makedirs("models", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Model A — binary classification metrics
# ══════════════════════════════════════════════════════════════════════════════

def eval_model_a_binary(X_test, y_test, clf) -> dict:
    print("Evaluating Model A binary classification …")
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    mf1    = f1_score(y_test, y_pred, average="macro")
    cm     = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {mf1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(classification_report(y_test, y_pred,
                                target_names=["incorrect", "correct"]))
    return {
        "accuracy":         round(float(acc), 4),
        "macro_f1":         round(float(mf1), 4),
        "confusion_matrix": cm.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Model A — Exact Match
# ══════════════════════════════════════════════════════════════════════════════

def eval_exact_match(test_df: pd.DataFrame, vectorizer, clf,
                     n_samples: int = 500) -> float:
    print(f"Computing Exact Match (n={n_samples}) …")
    sample  = test_df.sample(min(n_samples, len(test_df)), random_state=42)
    correct = 0
    for _, row in sample.iterrows():
        options = {l: clean_text(str(row[l])) for l in ["A", "B", "C", "D"]}
        pred, _ = predict_answer(clean_text(row["article"]),
                                 clean_text(row["question"]),
                                 options, vectorizer, clf)
        if pred == row["answer"]:
            correct += 1
    em = correct / len(sample)
    print(f"  Exact Match : {em:.4f}  ({correct}/{len(sample)})")
    return round(em, 4)


# ══════════════════════════════════════════════════════════════════════════════
# Model A — BLEU / ROUGE / METEOR
# ══════════════════════════════════════════════════════════════════════════════

def eval_model_a_generation(test_df: pd.DataFrame, vectorizer, clf,
                            n_samples: int = 500) -> dict:
    print(f"Computing BLEU/ROUGE/METEOR for Model A (n={n_samples}) …")
    bleu_m   = hf_load("bleu")
    rouge_m  = hf_load("rouge")
    meteor_m = hf_load("meteor")

    sample      = test_df.sample(min(n_samples, len(test_df)), random_state=42)
    preds, refs = [], []

    for _, row in sample.iterrows():
        options  = {l: clean_text(str(row[l])) for l in ["A", "B", "C", "D"]}
        pred_lbl, _ = predict_answer(clean_text(row["article"]),
                                     clean_text(row["question"]),
                                     options, vectorizer, clf)
        preds.append(str(row[pred_lbl]))
        refs.append(str(row[row["answer"]]))

    b = bleu_m.compute(predictions=preds,   references=[[r] for r in refs])
    r = rouge_m.compute(predictions=preds,  references=refs)
    m = meteor_m.compute(predictions=preds, references=refs)

    scores = {
        "bleu":    round(b["bleu"], 4),
        "rouge_l": round(r["rougeL"], 4),
        "meteor":  round(m["meteor"], 4),
    }
    print(f"  Model A generation scores: {scores}")
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# Unsupervised metrics (KMeans + LabelProp already saved by model_a_train.py)
# ══════════════════════════════════════════════════════════════════════════════

def _cluster_purity(true_labels, cluster_labels):
    total = 0
    for cid in np.unique(cluster_labels):
        mask     = cluster_labels == cid
        majority = mode(true_labels[mask], keepdims=True).mode[0]
        total   += np.sum(true_labels[mask] == majority)
    return total / len(true_labels)


def eval_unsupervised(X_test_ohe, y_test) -> dict:
    print("Evaluating unsupervised models …")
    svd_path = os.path.join("models", "model_a", "traditional", "svd_reducer.pkl")
    km_path  = os.path.join("models", "model_a", "traditional", "kmeans_model.pkl")
    lp_path  = os.path.join("models", "model_a", "traditional", "label_prop_model.pkl")

    svd = load_checkpoint(svd_path)
    km  = load_checkpoint(km_path)
    lp  = load_checkpoint(lp_path)

    results = {"kmeans_silhouette": None, "kmeans_purity": None, "label_prop_f1": None}

    if svd is not None and km is not None:
        X_r      = svd.transform(X_test_ohe)
        km_preds = km.predict(X_r)
        sil      = silhouette_score(X_r, km_preds, sample_size=5000, random_state=42)
        purity   = _cluster_purity(y_test, km_preds)
        results["kmeans_silhouette"] = round(float(sil), 4)
        results["kmeans_purity"]     = round(float(purity), 4)
        print(f"  KMeans  Silhouette={sil:.4f}  Purity={purity:.4f}")

        if lp is not None:
            lp_preds = lp.predict(X_r)
            lp_f1    = f1_score(y_test, lp_preds, average="macro")
            results["label_prop_f1"] = round(float(lp_f1), 4)
            print(f"  LabelProp  Macro F1={lp_f1:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Model B — distractor evaluation
# ══════════════════════════════════════════════════════════════════════════════

def eval_model_b_distractors(test_df: pd.DataFrame, dist_vec,
                             ranker=None, n_samples: int = 100) -> dict:
    print(f"Evaluating Model B distractors (n={n_samples}) …")
    bleu_m   = hf_load("bleu")
    rouge_m  = hf_load("rouge")
    meteor_m = hf_load("meteor")

    sample = test_df.sample(min(n_samples, len(test_df)), random_state=42)
    all_preds, all_refs = [], []
    hits, total         = 0, 0

    for _, row in sample.iterrows():
        article      = clean_text(row["article"])
        correct_text = clean_text(str(row[row["answer"]]))
        ref_dists    = [clean_text(str(row[l]))
                        for l in ["A", "B", "C", "D"] if l != row["answer"]]
        generated    = generate_distractors(
            article, clean_text(row["question"]),
            correct_text, dist_vec, n=3, ranker=ranker
        )
        for gen, ref in zip(generated, ref_dists):
            all_preds.append(gen)
            all_refs.append(ref)
            total += 1
            if correct_text.lower()[:20] not in gen.lower():
                hits += 1

    if not all_preds:
        return {"distractor_bleu": 0.0, "distractor_rouge_l": 0.0,
                "distractor_meteor": 0.0, "distractor_accuracy": 0.0}

    b = bleu_m.compute(predictions=all_preds,   references=[[r] for r in all_refs])
    r = rouge_m.compute(predictions=all_preds,  references=all_refs)
    m = meteor_m.compute(predictions=all_preds, references=all_refs)

    scores = {
        "distractor_bleu":     round(b["bleu"], 4),
        "distractor_rouge_l":  round(r["rougeL"], 4),
        "distractor_meteor":   round(m["meteor"], 4),
        "distractor_accuracy": round(hits / total, 4) if total > 0 else 0.0,
    }
    print(f"  {scores}")
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# Model B — hint evaluation
# ══════════════════════════════════════════════════════════════════════════════

def eval_model_b_hints(test_df: pd.DataFrame, dist_vec,
                       n_samples: int = 100) -> dict:
    print(f"Evaluating Model B hints (n={n_samples}) …")
    rouge_m  = hf_load("rouge")
    meteor_m = hf_load("meteor")

    sample = test_df.sample(min(n_samples, len(test_df)), random_state=42)
    all_hint3, all_refs = [], []

    for _, row in sample.iterrows():
        hints = generate_hints(
            clean_text(row["article"]),
            clean_text(row["question"]),
            clean_text(str(row[row["answer"]])),
            dist_vec,
        )
        all_hint3.append(hints["hint_3"])
        all_refs.append(clean_text(str(row[row["answer"]])))

    if not all_hint3:
        return {"hint_rouge_l": 0.0, "hint_meteor": 0.0}

    r = rouge_m.compute(predictions=all_hint3,  references=all_refs)
    m = meteor_m.compute(predictions=all_hint3, references=all_refs)
    scores = {"hint_rouge_l": round(r["rougeL"], 4),
              "hint_meteor":  round(m["meteor"], 4)}
    print(f"  {scores}")
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# Full evaluation — produces eval_results.json
# ══════════════════════════════════════════════════════════════════════════════

def run_full_evaluation(n_samples_a: int = 500, n_samples_b: int = 100) -> dict:
    # ── Load artefacts ────────────────────────────────────────────────────────
    models_dict = load_all_models()
    ohe_vec     = models_dict["ohe_vectorizer"]
    clf         = models_dict["clf"]
    model_used  = models_dict["model_used"]
    dist_vec    = models_dict["distractor_vectorizer"] or ohe_vec
    ranker      = models_dict["distractor_ranker"]

    test_df   = pd.read_csv(os.path.join("data", "raw", "test.csv"))
    X_test    = sp.load_npz(os.path.join("data", "processed", "X_test_full.npz"))
    X_test_ohe = sp.load_npz(os.path.join("data", "processed", "X_test_ohe.npz"))
    y_test    = np.load(os.path.join("data", "processed", "y_test.npy"))

    # ── Model A ───────────────────────────────────────────────────────────────
    binary_m = eval_model_a_binary(X_test, y_test, clf)
    gen_m    = eval_model_a_generation(test_df, ohe_vec, clf, n_samples=n_samples_a)
    em       = eval_exact_match(test_df, ohe_vec, clf, n_samples=n_samples_a)

    model_a = {
        "accuracy":         binary_m["accuracy"],
        "macro_f1":         binary_m["macro_f1"],
        "confusion_matrix": binary_m["confusion_matrix"],
        "bleu":             gen_m["bleu"],
        "rouge_l":          gen_m["rouge_l"],
        "meteor":           gen_m["meteor"],
        "exact_match":      em,
        "model_used":       model_used,
    }

    # ── Unsupervised ──────────────────────────────────────────────────────────
    unsupervised = eval_unsupervised(X_test_ohe, y_test)

    # ── Model B ───────────────────────────────────────────────────────────────
    dist_m = eval_model_b_distractors(test_df, dist_vec, ranker, n_samples=n_samples_b)
    hint_m = eval_model_b_hints(test_df, dist_vec, n_samples=n_samples_b)

    model_b = {
        "distractor_bleu":     dist_m["distractor_bleu"],
        "distractor_rouge_l":  dist_m["distractor_rouge_l"],
        "distractor_meteor":   dist_m["distractor_meteor"],
        "distractor_accuracy": dist_m["distractor_accuracy"],
        "hint_rouge_l":        hint_m["hint_rouge_l"],
        "hint_meteor":         hint_m["hint_meteor"],
        "n_samples_evaluated": n_samples_b,
    }

    # ── Benchmark reference (not trained here — cited values only) ────────────
    benchmark = {
        "bert_base_accuracy":  0.664,
        "bert_large_accuracy": 0.722,
        "t5_base_accuracy":    0.750,
        "source":              "Devlin et al. 2019, Raffel et al. 2020",
    }

    results = {
        "model_a":      model_a,
        "model_b":      model_b,
        "unsupervised": unsupervised,
        "benchmark":    benchmark,
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    _print_summary(results)
    print(f"\n✅ Evaluation complete.  Results saved to {RESULT_PATH}")
    return results


def _print_summary(results: dict) -> None:
    a   = results["model_a"]
    b   = results["model_b"]
    u   = results["unsupervised"]
    ref = results["benchmark"]
    print("\n" + "="*58)
    print("  FINAL RESULTS SUMMARY")
    print("="*58)
    print(f"  Model A  Accuracy    : {a.get('accuracy')}")
    print(f"  Model A  Macro F1    : {a.get('macro_f1')}")
    print(f"  Model A  Exact Match : {a.get('exact_match')}  (classifier: {a.get('model_used')})")
    print(f"  Model A  BLEU        : {a.get('bleu')}")
    print(f"  Model A  ROUGE-L     : {a.get('rouge_l')}")
    print(f"  Model A  METEOR      : {a.get('meteor')}")
    print(f"\n  Unsupervised  KMeans Silhouette : {u.get('kmeans_silhouette')}")
    print(f"  Unsupervised  KMeans Purity     : {u.get('kmeans_purity')}")
    print(f"  Unsupervised  LabelProp F1      : {u.get('label_prop_f1')}")
    print(f"\n  Model B  Dist BLEU   : {b.get('distractor_bleu')}")
    print(f"  Model B  Dist RG-L   : {b.get('distractor_rouge_l')}")
    print(f"  Model B  Dist METEOR : {b.get('distractor_meteor')}")
    print(f"  Model B  Dist Acc    : {b.get('distractor_accuracy')}")
    print(f"  Model B  Hint RG-L   : {b.get('hint_rouge_l')}")
    print(f"  Model B  Hint METEOR : {b.get('hint_meteor')}")
    print(f"\n  BERT-base  (reference) : {ref['bert_base_accuracy']:.1%}")
    print(f"  BERT-large (reference) : {ref['bert_large_accuracy']:.1%}")
    print(f"  T5-base    (reference) : {ref['t5_base_accuracy']:.1%}")


if __name__ == "__main__":
    run_full_evaluation(n_samples_a=500, n_samples_b=100)