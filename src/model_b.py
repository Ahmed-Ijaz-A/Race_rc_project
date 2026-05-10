# src/model_b_train.py
# ── COLAB SETUP (run these in a Colab cell before executing this script) ──────
# !git clone https://github.com/YOUR_USERNAME/race_rc_project.git /content/race_rc_project
# import os; os.chdir('/content/race_rc_project')
# !pip install -r requirements.txt -q
# ── Then run: python src/model_b_train.py ─────────────────────────────────────

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from evaluate import load as hf_load

from src.preprocessing import (clean_text, split_sentences,
                                save_checkpoint, load_checkpoint)

MODEL_DIR_B = os.path.join("models", "model_b", "traditional")
os.makedirs(MODEL_DIR_B, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Distractor vectorizer  (fit on article texts only — separate from Model A OHE)
# ══════════════════════════════════════════════════════════════════════════════

def get_distractor_vectorizer(train_df: pd.DataFrame):
    """
    CountVectorizer fitted on article texts only.
    Saved to models/model_b/traditional/distractor_vectorizer.pkl
    """
    path = os.path.join(MODEL_DIR_B, "distractor_vectorizer.pkl")
    vec  = load_checkpoint(path)
    if vec is not None:
        return vec

    print("Fitting distractor vectorizer on article texts …")
    articles = train_df["article"].apply(clean_text).tolist()
    vec = CountVectorizer(
        max_features=10_000,
        stop_words="english",
        binary=True,
        min_df=2,
        max_df=0.95,
    )
    vec.fit(articles)
    save_checkpoint(vec, path)
    return vec


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6 — Distractor generation helpers
# ══════════════════════════════════════════════════════════════════════════════

def extract_candidates(article: str, correct_answer: str,
                       vectorizer, top_n: int = 20) -> list:
    sentences = split_sentences(article)
    if not sentences:
        return []
    all_texts  = [correct_answer] + sentences
    vectors    = vectorizer.transform(all_texts)
    answer_vec = vectors[0]
    sent_vecs  = vectors[1:]
    sims       = cosine_similarity(answer_vec, sent_vecs)[0]
    ranked     = sorted(zip(sentences, sims), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def generate_distractors(article: str, question: str,
                         correct_answer: str, vectorizer,
                         n: int = 3, ranker=None) -> list:
    """
    Returns up to n distractor strings from article sentences.
    Applies diversity penalty (coeff=0.5) to avoid near-identical picks.
    Explicit filter: skip any candidate where correct_answer[:30] appears verbatim.
    """
    candidates = extract_candidates(article, correct_answer, vectorizer, top_n=20)
    filtered   = [
        (s, sim) for s, sim in candidates
        if correct_answer.lower()[:30] not in s.lower() and sim < 0.95
    ]
    if not filtered:
        filtered = candidates  # fallback

    if ranker is not None:
        texts      = [s for s, _ in filtered]
        v_art      = vectorizer.transform([article])
        v_cor      = vectorizer.transform([correct_answer])
        feat_rows  = []
        for sent in texts:
            v_s = vectorizer.transform([sent])
            feat_rows.append([
                cosine_similarity(v_s, v_art)[0][0],
                cosine_similarity(v_s, v_cor)[0][0],
                len(sent.split()) / max(1, len(correct_answer.split())),
            ])
        probs    = ranker.predict_proba(np.array(feat_rows))[:, 1]
        order    = np.argsort(probs)[::-1]
        filtered = [filtered[i] for i in order]

    candidate_pool = [s for s, _ in filtered]
    selected: list = []
    for _ in range(n):
        if not candidate_pool:
            break
        scores = []
        for cand in candidate_pool:
            v1       = vectorizer.transform([cand])
            v2       = vectorizer.transform([correct_answer])
            base_sim = cosine_similarity(v1, v2)[0][0]
            div      = sum(cosine_similarity(v1, vectorizer.transform([s]))[0][0]
                           for s in selected)
            scores.append(base_sim - 0.5 * div)
        selected.append(candidate_pool.pop(int(np.argmax(scores))))

    return selected[:n]


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6.3 — Distractor ranker (LogisticRegression on 3 cosine/length features)
# ══════════════════════════════════════════════════════════════════════════════

def build_distractor_features(df: pd.DataFrame, vectorizer) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        correct_text = clean_text(str(row[row["answer"]]))
        article_text = clean_text(row["article"])
        for lbl in ["A", "B", "C", "D"]:
            if lbl == row["answer"]:
                continue
            opt  = clean_text(str(row[lbl]))
            v_o  = vectorizer.transform([opt])
            v_a  = vectorizer.transform([article_text])
            v_c  = vectorizer.transform([correct_text])
            rows.append({
                "sim_to_article": cosine_similarity(v_o, v_a)[0][0],
                "sim_to_correct": cosine_similarity(v_o, v_c)[0][0],
                "len_ratio":      len(opt.split()) / max(1, len(correct_text.split())),
                "label":          1,
            })
    return pd.DataFrame(rows)


def train_distractor_ranker(train_df: pd.DataFrame, vectorizer) -> LogisticRegression:
    path   = os.path.join(MODEL_DIR_B, "distractor_ranker.pkl")
    ranker = load_checkpoint(path)
    if ranker is not None:
        return ranker

    print("Building distractor training features …")
    sample  = train_df.sample(min(5000, len(train_df)), random_state=42)
    feat_df = build_distractor_features(sample, vectorizer)
    X = feat_df[["sim_to_article", "sim_to_correct", "len_ratio"]].values
    y = feat_df["label"].values

    print("Training distractor ranker …")
    ranker = LogisticRegression(max_iter=200, random_state=42)
    ranker.fit(X, y)
    save_checkpoint(ranker, path)
    return ranker


# ══════════════════════════════════════════════════════════════════════════════
# Phase 7 — Hint generation + hint scorer
# ══════════════════════════════════════════════════════════════════════════════

def generate_hints(article: str, question: str,
                   correct_answer: str, vectorizer) -> dict:
    """
    Returns {'hint_1': str, 'hint_2': str, 'hint_3': str}
    hint_1 = lowest relevance (general), hint_3 = highest relevance (near answer).
    hint_3 never directly states the correct answer verbatim.
    """
    sentences = split_sentences(article)
    while len(sentences) < 3:
        sentences.append(sentences[-1] if sentences else "No hint available.")

    texts  = [question] + sentences
    vecs   = vectorizer.transform(texts)
    q_vec  = vecs[0]
    s_vecs = vecs[1:]
    sims   = cosine_similarity(q_vec, s_vecs)[0]
    ranked = sorted(zip(sentences, sims), key=lambda x: x[1], reverse=True)

    filtered = [(s, sim) for s, sim in ranked
                if correct_answer.lower()[:20] not in s.lower()]
    if len(filtered) < 3:
        filtered = ranked  # fallback

    n = len(filtered)
    return {
        "hint_1": filtered[n - 1][0],
        "hint_2": filtered[n // 2][0],
        "hint_3": filtered[0][0],
    }


def build_hint_features(article: str, question: str,
                        sentence: str, vectorizer,
                        position_idx: int, total: int) -> list:
    """5-feature vector for a single sentence candidate."""
    v_s = vectorizer.transform([sentence])
    v_q = vectorizer.transform([question])
    v_a = vectorizer.transform([article])
    return [
        cosine_similarity(v_s, v_q)[0][0],
        cosine_similarity(v_s, v_a)[0][0],
        len(sentence.split()),
        position_idx / max(1, total),
        len(set(sentence.lower().split()) & set(question.lower().split())),
    ]


def train_hint_scorer(train_df: pd.DataFrame, vectorizer) -> LogisticRegression:
    """
    Trains a LogisticRegression to score hint-worthiness of a sentence.
    Positive label: sentence with highest cosine similarity to question.
    Negative label: sentences with lowest cosine similarity.
    Saved to models/model_b/traditional/hint_scorer.pkl
    """
    path   = os.path.join(MODEL_DIR_B, "hint_scorer.pkl")
    scorer = load_checkpoint(path)
    if scorer is not None:
        return scorer

    print("Building hint scorer training data …")
    sample = train_df.sample(min(3000, len(train_df)), random_state=42)
    rows   = []

    for _, row in sample.iterrows():
        article  = clean_text(row["article"])
        question = clean_text(row["question"])
        correct  = clean_text(str(row[row["answer"]]))
        sents    = split_sentences(article)
        if len(sents) < 2:
            continue

        vecs  = vectorizer.transform([question] + sents)
        q_vec = vecs[0]
        s_vecs = vecs[1:]
        sims  = cosine_similarity(q_vec, s_vecs)[0]

        # Most similar = positive hint; least similar = negative hint
        best_idx  = int(np.argmax(sims))
        worst_idx = int(np.argmin(sims))

        for i, (sent, sim) in enumerate(zip(sents, sims)):
            label = 1 if i == best_idx else (0 if i == worst_idx else -1)
            if label == -1:
                continue
            feats = build_hint_features(article, question, sent,
                                        vectorizer, i, len(sents))
            rows.append(feats + [label])

    feat_df = pd.DataFrame(rows, columns=["sim_q", "sim_art", "length",
                                           "position", "word_overlap", "label"])
    X = feat_df[["sim_q", "sim_art", "length", "position", "word_overlap"]].values
    y = feat_df["label"].values

    print(f"Training hint scorer on {len(X)} samples …")
    scorer = LogisticRegression(max_iter=200, random_state=42)
    scorer.fit(X, y)
    save_checkpoint(scorer, path)
    return scorer


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6.4 + 7 — Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_distractors(test_df: pd.DataFrame, vectorizer,
                         ranker=None, n_samples: int = 100) -> dict:
    bleu_m   = hf_load("bleu")
    rouge_m  = hf_load("rouge")
    meteor_m = hf_load("meteor")

    sample = test_df.sample(min(n_samples, len(test_df)), random_state=42)
    all_preds, all_refs = [], []
    hits, total         = 0, 0

    for _, row in sample.iterrows():
        article      = clean_text(row["article"])
        correct_text = clean_text(str(row[row["answer"]]))
        ref_distractors = [clean_text(str(row[l]))
                           for l in ["A", "B", "C", "D"] if l != row["answer"]]
        generated = generate_distractors(
            article, clean_text(row["question"]),
            correct_text, vectorizer, n=3, ranker=ranker
        )
        for gen, ref in zip(generated, ref_distractors):
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
    print(f"\nDistractor evaluation (n={total}): {scores}")
    return scores


def evaluate_hints(test_df: pd.DataFrame, vectorizer, n_samples: int = 100) -> dict:
    rouge_m  = hf_load("rouge")
    meteor_m = hf_load("meteor")

    sample = test_df.sample(min(n_samples, len(test_df)), random_state=42)
    all_hint3, all_refs = [], []

    for _, row in sample.iterrows():
        hints = generate_hints(
            clean_text(row["article"]),
            clean_text(row["question"]),
            clean_text(str(row[row["answer"]])),
            vectorizer,
        )
        all_hint3.append(hints["hint_3"])
        all_refs.append(clean_text(str(row[row["answer"]])))

    if not all_hint3:
        return {"hint_rouge_l": 0.0, "hint_meteor": 0.0}

    r = rouge_m.compute(predictions=all_hint3,  references=all_refs)
    m = meteor_m.compute(predictions=all_hint3, references=all_refs)
    scores = {"hint_rouge_l": round(r["rougeL"], 4),
              "hint_meteor":  round(m["meteor"], 4)}
    print(f"Hint evaluation scores: {scores}")
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    train_df = pd.read_csv(os.path.join("data", "raw", "train.csv"))
    test_df  = pd.read_csv(os.path.join("data", "raw", "test.csv"))

    # ── Distractor vectorizer (fit on articles only) ─────────────────────────
    dist_vec = get_distractor_vectorizer(train_df)

    # ── Distractor ranker ────────────────────────────────────────────────────
    ranker = train_distractor_ranker(train_df, dist_vec)

    # ── Hint scorer ──────────────────────────────────────────────────────────
    scorer = train_hint_scorer(train_df, dist_vec)

    # ── Evaluation ───────────────────────────────────────────────────────────
    dist_scores = evaluate_distractors(test_df, dist_vec, ranker=ranker, n_samples=100)
    hint_scores = evaluate_hints(test_df, dist_vec, n_samples=100)

    # ── Sample outputs ───────────────────────────────────────────────────────
    print("\n── 3 Sample outputs ──")
    for _, row in test_df.sample(3, random_state=7).iterrows():
        article  = clean_text(row["article"])
        q        = clean_text(row["question"])
        correct  = clean_text(str(row[row["answer"]]))
        dists    = generate_distractors(article, q, correct, dist_vec, n=3, ranker=ranker)
        hints    = generate_hints(article, q, correct, dist_vec)
        print(f"\nQ: {row['question'][:80]}")
        print(f"  Correct     : {str(row[row['answer']])[:60]}")
        print(f"  Distractors : {dists}")
        print(f"  Hint 1      : {hints['hint_1'][:80]}")
        print(f"  Hint 3      : {hints['hint_3'][:80]}")

    # ── Save results JSON ────────────────────────────────────────────────────
    results = {**dist_scores, **hint_scores, "n_samples_evaluated": 100}
    os.makedirs("models", exist_ok=True)
    with open(os.path.join("models", "model_b_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ model_b_train.py complete.")
    print("   Saved → models/model_b/traditional/distractor_vectorizer.pkl")
    print("   Saved → models/model_b/traditional/distractor_ranker.pkl")
    print("   Saved → models/model_b/traditional/hint_scorer.pkl")
    print("   Results → models/model_b_results.json")


if __name__ == "__main__":
    main()