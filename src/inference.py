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


def _article_sentence_candidates(article: str) -> list:
    """
    Return unique article sentences as (display_text, cleaned_text).

    Sentence splitting must happen before clean_text(), because clean_text()
    removes punctuation that split_sentences() uses to find boundaries.
    """
    raw_sentences = split_sentences(article)
    if not raw_sentences and str(article).strip():
        raw_sentences = [str(article).strip()]

    candidates = []
    seen = set()
    for sentence in raw_sentences:
        display = " ".join(str(sentence).split())
        cleaned = clean_text(display)
        if len(cleaned) < 10 or cleaned in seen:
            continue
        seen.add(cleaned)
        candidates.append((display, cleaned))
    return candidates


def _too_similar(text: str, selected: list, vectorizer, threshold: float = 0.88) -> bool:
    if not selected:
        return False
    text_vec = vectorizer.transform([text])
    selected_vecs = vectorizer.transform(selected)
    return bool(np.max(cosine_similarity(text_vec, selected_vecs)[0]) >= threshold)


# ══════════════════════════════════════════════════════════════════════════════
# Model B — distractor generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_distractors(article: str, question: str,
                         correct_answer: str, vectorizer,
                         n: int = 3, ranker=None,
                         exclude_texts: list = None) -> list:
    """
    Return up to n distractor strings extracted from the article.
    Uses the distractor_vectorizer (not ohe_vectorizer).

    Guarantees:
      - No distractor contains correct_answer[:30] verbatim
      - Diversity penalty (coeff=0.5) prevents near-identical picks
    """
    art     = clean_text(article)
    correct = clean_text(correct_answer)
    excluded = {clean_text(text) for text in (exclude_texts or []) if str(text).strip()}

    sentence_pairs = _article_sentence_candidates(article)
    if not sentence_pairs:
        return []

    sentence_texts = [cleaned for _, cleaned in sentence_pairs]
    all_texts  = [correct] + sentence_texts
    vectors    = vectorizer.transform(all_texts)
    ans_vec    = vectors[0]
    sent_vecs  = vectors[1:]
    sims       = cosine_similarity(ans_vec, sent_vecs)[0]
    ranked     = sorted(
        [(display, cleaned, sim) for (display, cleaned), sim in zip(sentence_pairs, sims)],
        key=lambda x: x[2],
        reverse=True,
    )[:20]

    filtered = [
        (display, cleaned, sim) for display, cleaned, sim in ranked
        if correct[:30] not in cleaned and cleaned not in excluded and sim < 0.95
    ]
    if not filtered:
        filtered = [
            (display, cleaned, sim) for display, cleaned, sim in ranked
            if cleaned not in excluded
        ]

    if ranker is not None:
        v_art     = vectorizer.transform([art])
        v_cor     = vectorizer.transform([correct])
        feat_rows = []
        for _, cleaned, _ in filtered:
            v_s = vectorizer.transform([cleaned])
            feat_rows.append([
                cosine_similarity(v_s, v_art)[0][0],
                cosine_similarity(v_s, v_cor)[0][0],
                len(cleaned.split()) / max(1, len(correct.split())),
            ])
        probs    = ranker.predict_proba(np.array(feat_rows))[:, 1]
        order    = np.argsort(probs)[::-1]
        filtered = [filtered[i] for i in order]

    pool:     list = [(display, cleaned) for display, cleaned, _ in filtered]
    selected: list = []
    selected_cleaned: list = []
    for _ in range(n):
        if not pool:
            break
        scores = []
        for _, cand_cleaned in pool:
            if _too_similar(cand_cleaned, selected_cleaned, vectorizer):
                scores.append(-np.inf)
                continue
            v1       = vectorizer.transform([cand_cleaned])
            v2       = vectorizer.transform([correct])
            base_sim = cosine_similarity(v1, v2)[0][0]
            div      = sum(cosine_similarity(v1, vectorizer.transform([s]))[0][0]
                           for s in selected_cleaned)
            scores.append(base_sim - 0.5 * div)
        best_idx = int(np.argmax(scores))
        if not np.isfinite(scores[best_idx]):
            break
        best_display, best_cleaned = pool.pop(best_idx)
        selected.append(best_display)
        selected_cleaned.append(best_cleaned)

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

    sentence_pairs = _article_sentence_candidates(article)
    if not sentence_pairs:
        return {
            "hint_1": "Review the question and compare it with the passage.",
            "hint_2": "Eliminate options not supported by the passage.",
            "hint_3": "Look for the sentence most directly related to the question.",
        }

    sentence_texts = [cleaned for _, cleaned in sentence_pairs]
    texts  = [q] + sentence_texts
    vecs   = vectorizer.transform(texts)
    q_vec  = vecs[0]
    s_vecs = vecs[1:]
    sims   = cosine_similarity(q_vec, s_vecs)[0]
    ranked = sorted(
        [(display, cleaned, sim) for (display, cleaned), sim in zip(sentence_pairs, sims)],
        key=lambda x: x[2],
        reverse=True,
    )

    filtered = [(display, cleaned, sim) for display, cleaned, sim in ranked
                if correct[:20] not in cleaned]
    if len(filtered) < 3:
        filtered = filtered + [item for item in ranked if item not in filtered]

    pick_order = [len(filtered) - 1, len(filtered) // 2, 0]
    selected = []
    selected_cleaned = []
    for idx in pick_order:
        display, cleaned, _ = filtered[idx]
        if cleaned not in selected_cleaned and not _too_similar(
            cleaned, selected_cleaned, vectorizer, threshold=0.75
        ):
            selected.append(display)
            selected_cleaned.append(cleaned)

    for display, cleaned, _ in filtered:
        if len(selected) == 3:
            break
        if cleaned in selected_cleaned or _too_similar(
            cleaned, selected_cleaned, vectorizer, threshold=0.75
        ):
            continue
        selected.append(display)
        selected_cleaned.append(cleaned)

    fallback_hints = [
        "Review the question and compare it with the passage.",
        "Eliminate options not supported by the passage.",
        "Look for the sentence most directly related to the question.",
    ]
    for hint in fallback_hints:
        if len(selected) == 3:
            break
        selected.append(hint)

    return {
        "hint_1": selected[0],
        "hint_2": selected[1],
        "hint_3": selected[2],
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
