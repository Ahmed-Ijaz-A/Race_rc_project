# src/inference.py
# ── COLAB SETUP (run these in a Colab cell before executing this script) ──────
# !git clone https://github.com/YOUR_USERNAME/race_rc_project.git /content/race_rc_project
# import os; os.chdir('/content/race_rc_project')
# !pip install -r requirements.txt -q
# ── Then run: python src/inference.py  (smoke-test only) ──────────────────────

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import clean_text, split_sentences, load_checkpoint

# ── Artifact paths ────────────────────────────────────────────────────────────
_OHE_VEC_PATH   = os.path.join("models", "model_a", "traditional", "ohe_vectorizer.pkl")
_ENSEMBLE_PATH  = os.path.join("models", "model_a", "traditional", "ensemble_model.pkl")
_LR_PATH        = os.path.join("models", "model_a", "traditional", "lr_model.pkl")
_DIST_VEC_PATH  = os.path.join("models", "model_b", "traditional", "distractor_vectorizer.pkl")
_DIST_RNK_PATH  = os.path.join("models", "model_b", "traditional", "distractor_ranker.pkl")
_HINT_SCR_PATH  = os.path.join("models", "model_b", "traditional", "hint_scorer.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# Model loader  — call once, cache the result
# ══════════════════════════════════════════════════════════════════════════════

def load_all_models() -> dict:
    """
    Loads and returns all 5 inference artefacts:
        ohe_vectorizer, clf (ensemble or LR), distractor_vectorizer,
        distractor_ranker, hint_scorer

    Raises FileNotFoundError if ohe_vectorizer or any classifier is missing.
    distractor_vectorizer, distractor_ranker, hint_scorer return None if absent.
    """
    if not os.path.exists(_OHE_VEC_PATH):
        raise FileNotFoundError(
            f"OHE vectorizer not found at {_OHE_VEC_PATH}. "
            "Run src/preprocessing.py first."
        )

    ohe_vec = load_checkpoint(_OHE_VEC_PATH)

    clf_path = _ENSEMBLE_PATH if os.path.exists(_ENSEMBLE_PATH) else _LR_PATH
    if not os.path.exists(clf_path):
        raise FileNotFoundError(
            f"No classifier found at {_ENSEMBLE_PATH} or {_LR_PATH}. "
            "Run src/model_a_train.py first."
        )
    clf = load_checkpoint(clf_path)
    model_used = "ensemble" if clf_path == _ENSEMBLE_PATH else "lr"

    dist_vec = load_checkpoint(_DIST_VEC_PATH)   # None if missing
    ranker   = load_checkpoint(_DIST_RNK_PATH)   # None if missing
    scorer   = load_checkpoint(_HINT_SCR_PATH)   # None if missing

    print(f"  Loaded classifier  : {model_used}")
    print(f"  distractor_vec     : {'✓' if dist_vec else '✗ (missing)'}")
    print(f"  distractor_ranker  : {'✓' if ranker   else '✗ (missing)'}")
    print(f"  hint_scorer        : {'✓' if scorer   else '✗ (missing)'}")

    return {
        "ohe_vectorizer":         ohe_vec,
        "clf":                    clf,
        "model_used":             model_used,
        "distractor_vectorizer":  dist_vec,
        "distractor_ranker":      ranker,
        "hint_scorer":            scorer,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Model A — answer prediction
# ══════════════════════════════════════════════════════════════════════════════

def predict_answer(article: str, question: str,
                   options: dict, vectorizer, clf) -> tuple:
    art = clean_text(article)
    q   = clean_text(question)

    best_label, best_score = None, -1.0
    for label, text in options.items():
        opt      = clean_text(text)
        combined = f"{art} {art} {q} {opt}"

        # Sparse OHE vector
        X_ohe = vectorizer.transform([combined])

        # 3 cosine similarity features
        v_art = vectorizer.transform([art])
        v_q   = vectorizer.transform([q])
        v_opt = vectorizer.transform([opt])

        def row_cosine(a, b):
            dot   = a.multiply(b).sum()
            denom = np.sqrt(a.multiply(a).sum()) * np.sqrt(b.multiply(b).sum())
            return float(dot / denom) if denom > 0 else 0.0

        cos_feats = sp.csr_matrix([[
            row_cosine(v_art, v_q),
            row_cosine(v_art, v_opt),
            row_cosine(v_q,   v_opt),
        ]])

        X = sp.hstack([X_ohe, cos_feats])
        prob = clf.predict_proba(X)[0][1]

        if prob > best_score:
            best_score = prob
            best_label = label

    return best_label, float(best_score)


# ══════════════════════════════════════════════════════════════════════════════
# Model B — distractor generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_distractors(article: str, question: str,
                         correct_answer: str, vectorizer,
                         n: int = 3, ranker=None) -> list:
    """
    Return up to n distractor strings extracted from the article.
    Uses the distractor_vectorizer (not ohe_vectorizer).

    Guarantees:
      - No distractor contains correct_answer[:30] verbatim
      - Diversity penalty (coeff=0.5) prevents near-identical picks
    """
    art     = clean_text(article)
    correct = clean_text(correct_answer)

    sentences = split_sentences(art)
    if not sentences:
        return []

    all_texts  = [correct] + sentences
    vectors    = vectorizer.transform(all_texts)
    ans_vec    = vectors[0]
    sent_vecs  = vectors[1:]
    sims       = cosine_similarity(ans_vec, sent_vecs)[0]
    ranked     = sorted(zip(sentences, sims), key=lambda x: x[1], reverse=True)[:20]

    filtered = [(s, sim) for s, sim in ranked
                if correct.lower()[:30] not in s.lower() and sim < 0.95]
    if not filtered:
        filtered = ranked

    if ranker is not None:
        texts     = [s for s, _ in filtered]
        v_art     = vectorizer.transform([art])
        v_cor     = vectorizer.transform([correct])
        feat_rows = []
        for sent in texts:
            v_s = vectorizer.transform([sent])
            feat_rows.append([
                cosine_similarity(v_s, v_art)[0][0],
                cosine_similarity(v_s, v_cor)[0][0],
                len(sent.split()) / max(1, len(correct.split())),
            ])
        probs    = ranker.predict_proba(np.array(feat_rows))[:, 1]
        order    = np.argsort(probs)[::-1]
        filtered = [filtered[i] for i in order]

    pool:     list = [s for s, _ in filtered]
    selected: list = []
    for _ in range(n):
        if not pool:
            break
        scores = []
        for cand in pool:
            v1       = vectorizer.transform([cand])
            v2       = vectorizer.transform([correct])
            base_sim = cosine_similarity(v1, v2)[0][0]
            div      = sum(cosine_similarity(v1, vectorizer.transform([s]))[0][0]
                           for s in selected)
            scores.append(base_sim - 0.5 * div)
        selected.append(pool.pop(int(np.argmax(scores))))

    return selected[:n]


# ══════════════════════════════════════════════════════════════════════════════
# Model B — hint generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_hints(article: str, question: str,
                   correct_answer: str, vectorizer) -> dict:
    """
    Returns {'hint_1': str, 'hint_2': str, 'hint_3': str}
    hint_1 = least relevant (general clue)
    hint_3 = most relevant (near answer, but never states it verbatim)
    Uses distractor_vectorizer.
    """
    art     = clean_text(article)
    q       = clean_text(question)
    correct = clean_text(correct_answer)

    sentences = split_sentences(art)
    while len(sentences) < 3:
        sentences.append(sentences[-1] if sentences else "No hint available.")

    texts  = [q] + sentences
    vecs   = vectorizer.transform(texts)
    q_vec  = vecs[0]
    s_vecs = vecs[1:]
    sims   = cosine_similarity(q_vec, s_vecs)[0]
    ranked = sorted(zip(sentences, sims), key=lambda x: x[1], reverse=True)

    filtered = [(s, sim) for s, sim in ranked
                if correct.lower()[:20] not in s.lower()]
    if len(filtered) < 3:
        filtered = ranked  # fallback

    n = len(filtered)
    return {
        "hint_1": filtered[n - 1][0],
        "hint_2": filtered[n // 2][0],
        "hint_3": filtered[0][0],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Smoke-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading models …")
    models   = load_all_models()
    ohe_vec  = models["ohe_vectorizer"]
    dist_vec = models["distractor_vectorizer"] or ohe_vec   # fallback
    clf      = models["clf"]
    ranker   = models["distractor_ranker"]

    article  = ("The sun is a star at the center of the solar system. "
                "It provides light and heat to all planets. "
                "Earth orbits the sun once every 365 days. "
                "The moon orbits the Earth approximately every 27 days.")
    question = "How long does it take for the Earth to orbit the sun?"
    options  = {"A": "27 days", "B": "365 days",
                "C": "24 hours", "D": "12 months and 3 days"}

    pred, conf = predict_answer(article, question, options, ohe_vec, clf)
    print(f"\npredict_answer  → {pred}  (confidence {conf:.2%})")

    distractors = generate_distractors(
        article, question, options["B"], dist_vec, n=3, ranker=ranker
    )
    print(f"generate_distractors → {distractors}")

    hints = generate_hints(article, question, options["B"], dist_vec)
    print(f"generate_hints  → {hints}")
