# ui/app.py
# Run from project root: streamlit run ui/app.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os

# BASE = project root (one level up from ui/)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

import json
import pandas as pd
import streamlit as st

# st.set_page_config must be the absolute first Streamlit call
st.set_page_config(
    page_title="RACE Quiz Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

from preprocessing import clean_text, split_sentences
from inference import (predict_answer, generate_distractors,
                       generate_hints, load_all_models)


# ══════════════════════════════════════════════════════════════════════════════
# Cached loaders
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading models …")
def load_models():
    """
    Loads and returns all 5 artefacts:
        ohe_vectorizer, clf, distractor_vectorizer, distractor_ranker, hint_scorer
    """
    m = load_all_models()
    return (
        m["ohe_vectorizer"],
        m["clf"],
        m["distractor_vectorizer"],
        m["distractor_ranker"],
        m["hint_scorer"],
    )


@st.cache_data(show_spinner=False)
def load_race_sample() -> pd.DataFrame:
    path = os.path.join(BASE, "data", "raw", "test.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_eval_results() -> dict | None:
    path = os.path.join(BASE, "models", "eval_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── Initialise artefacts ──────────────────────────────────────────────────────
_models_ok  = False
_model_err  = ""
ohe_vec     = None
clf         = None
dist_vec    = None
ranker      = None
scorer      = None

try:
    ohe_vec, clf, dist_vec, ranker, scorer = load_models()
    # Fall back to ohe_vec for Model B if distractor_vec not yet trained
    if dist_vec is None:
        dist_vec = ohe_vec
    _models_ok = True
except Exception as e:
    _model_err = str(e)

race_df  = load_race_sample()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📚 RACE Quiz")
    st.caption("AI Course · BS(CS) Spring 2026")
    st.divider()
    screen = st.radio(
        "Navigate",
        ["Article Input", "Quiz", "Hints", "Analytics"],
        index=0,
    )
    st.divider()
    if not _models_ok:
        st.error(f"Model load failed:\n{_model_err}")
    else:
        st.success("Models loaded ✓")
    st.caption("⚠️ AI-generated — errors are possible.")


# ══════════════════════════════════════════════════════════════════════════════
# Screen 1 — Article Input
# ══════════════════════════════════════════════════════════════════════════════

if screen == "Article Input":
    st.header("Screen 1 — Article Input")

    col_rand, col_clear = st.columns([1, 1])

    with col_rand:
        if st.button("🎲 Load Random RACE Sample", disabled=race_df.empty):
            row = race_df.sample(1, random_state=None).iloc[0]
            st.session_state["article"]  = row["article"]
            st.session_state["question"] = row["question"]
            st.session_state["options"]  = {l: str(row[l]) for l in ["A", "B", "C", "D"]}
            st.session_state["answer"]   = row["answer"]
            for key in ["hints_revealed", "last_pred"]:
                st.session_state.pop(key, None)
            st.success("Random RACE sample loaded.  Switch to the Quiz screen.")

    with col_clear:
        if st.button("🗑️ Clear All"):
            for key in ["article", "question", "options", "answer",
                        "hints_revealed", "last_pred", "session_log"]:
                st.session_state.pop(key, None)
            st.rerun()

    st.divider()

    article_input = st.text_area(
        "Paste a reading passage:",
        value=st.session_state.get("article", ""),
        height=300,
        placeholder="Paste any English reading passage here …",
    )
    question_input = st.text_input(
        "Question (optional if using RACE sample):",
        value=st.session_state.get("question", ""),
        placeholder="e.g. What is the main idea of the passage?",
    )

    if st.button("Submit Article", type="primary"):
        if not article_input.strip():
            st.error("Please enter an article or load a random sample.")
        else:
            st.session_state["article"] = article_input.strip()
            if question_input.strip():
                st.session_state["question"] = question_input.strip()
            st.success("✅ Article saved.  Switch to the Quiz screen.")

    if "article" in st.session_state:
        with st.expander("Preview stored article (first 500 chars)"):
            st.write(st.session_state["article"][:500] + " …")


# ══════════════════════════════════════════════════════════════════════════════
# Screen 2 — Quiz
# ══════════════════════════════════════════════════════════════════════════════

elif screen == "Quiz":
    st.header("Screen 2 — Quiz")

    if "article" not in st.session_state:
        st.info("👈 Please load an article first from **Screen 1**.")
        st.stop()

    if not _models_ok:
        st.error("Models not loaded — check sidebar.")
        st.stop()

    question = st.session_state.get("question", "")
    if not question:
        question = st.text_input("Enter the question:")
        if question:
            st.session_state["question"] = question
    if not question:
        st.warning("No question found.  Enter one above or load a RACE sample.")
        st.stop()

    st.subheader(question)

    options = st.session_state.get("options", {})
    if not options:
        st.info("Enter the four answer options below:")
        opt_cols = st.columns(2)
        for i, lbl in enumerate(["A", "B", "C", "D"]):
            with opt_cols[i % 2]:
                val = st.text_input(f"Option {lbl}", key=f"opt_input_{lbl}")
                if val:
                    options[lbl] = val
        if len(options) == 4:
            st.session_state["options"] = options
            st.rerun()
        else:
            st.stop()

    user_answer = st.radio(
        "Choose your answer:",
        options=list(options.keys()),
        format_func=lambda x: f"**{x}.** {options[x]}",
    )

    if st.button("✅ Check Answer", type="primary"):
        with st.spinner("Verifying with Model A …"):
            art      = clean_text(st.session_state["article"])
            q        = clean_text(question)
            opts     = {k: clean_text(v) for k, v in options.items()}
            pred_lbl, conf = predict_answer(art, q, opts, ohe_vec, clf)

        st.session_state["last_pred"] = pred_lbl
        correct_label = st.session_state.get("answer", pred_lbl)

        if user_answer == correct_label:
            st.success(
                f"✅ Correct!  The answer is **{correct_label}. {options[correct_label]}**"
            )
        else:
            st.error(
                f"❌ Incorrect.  You chose **{user_answer}**.  "
                f"Correct answer: **{correct_label}. {options[correct_label]}**"
            )

        st.metric("Model A Confidence", f"{conf:.1%}")
        st.caption("⚠️ AI-verified — errors are possible.")

        if "session_log" not in st.session_state:
            st.session_state["session_log"] = []
        st.session_state["session_log"].append({
            "question":       question[:60],
            "user_answer":    user_answer,
            "correct_answer": correct_label,
            "model_pred":     pred_lbl,
            "confidence":     round(conf, 3),
            "correct":        user_answer == correct_label,
        })


# ══════════════════════════════════════════════════════════════════════════════
# Screen 3 — Hints
# ══════════════════════════════════════════════════════════════════════════════

elif screen == "Hints":
    st.header("Screen 3 — Hints")

    if "article" not in st.session_state or "question" not in st.session_state:
        st.info("👈 Please load an article and question from **Screen 1** first.")
        st.stop()

    if not _models_ok:
        st.error("Models not loaded — check sidebar.")
        st.stop()

    article      = clean_text(st.session_state["article"])
    question     = clean_text(st.session_state["question"])
    correct_lbl  = st.session_state.get("answer", "A")
    options      = st.session_state.get("options", {})
    correct_text = clean_text(str(options.get(correct_lbl, "")))

    # Compute hints once per article and cache in session
    hint_key = f"hints_{hash(article[:50])}"
    if hint_key not in st.session_state:
        with st.spinner("Generating hints …"):
            st.session_state[hint_key] = generate_hints(
                article, question, correct_text, dist_vec
            )
    hints = st.session_state[hint_key]

    st.write(f"**Question:** {st.session_state['question']}")
    st.divider()

    if "hints_revealed" not in st.session_state:
        st.session_state["hints_revealed"] = 0

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💡 Hint 1 (General)"):
            st.session_state["hints_revealed"] = max(
                st.session_state["hints_revealed"], 1
            )
    with col2:
        if st.button("🔍 Hint 2 (Specific)",
                     disabled=st.session_state["hints_revealed"] < 1):
            st.session_state["hints_revealed"] = max(
                st.session_state["hints_revealed"], 2
            )
    with col3:
        if st.button("🎯 Hint 3 (Near Answer)",
                     disabled=st.session_state["hints_revealed"] < 2):
            st.session_state["hints_revealed"] = max(
                st.session_state["hints_revealed"], 3
            )

    n = st.session_state["hints_revealed"]
    if n >= 1:
        with st.expander("💡 Hint 1", expanded=True):
            st.write(hints["hint_1"])
    if n >= 2:
        with st.expander("🔍 Hint 2", expanded=True):
            st.write(hints["hint_2"])
    if n >= 3:
        with st.expander("🎯 Hint 3", expanded=True):
            st.write(hints["hint_3"])
        st.divider()
        if st.button("🔓 Reveal Answer", type="primary"):
            ans_text = options.get(correct_lbl, "N/A")
            st.success(f"The correct answer is: **{correct_lbl}. {ans_text}**")

    st.divider()
    if st.button("🧩 Show generated distractors"):
        with st.spinner("Generating distractors …"):
            dists = generate_distractors(
                article, question, correct_text,
                dist_vec, n=3, ranker=ranker
            )
        st.subheader("Generated Distractors")
        for i, d in enumerate(dists, 1):
            st.write(f"{i}. {d}")
        st.caption("⚠️ AI-generated alternatives — review before use.")


# ══════════════════════════════════════════════════════════════════════════════
# Screen 4 — Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════════════

elif screen == "Analytics":
    st.header("Screen 4 — Analytics Dashboard")
    results = load_eval_results()

    if results is None:
        st.warning(
            "No evaluation results found at `models/eval_results.json`.  "
            "Run `python src/evaluate.py` from the project root to generate it."
        )
    else:
        a   = results.get("model_a", {})
        b   = results.get("model_b", {})
        u   = results.get("unsupervised", {})
        ref = results.get("benchmark", {})

        # ── Model A ────────────────────────────────────────────────────────────
        st.subheader("Model A — Answer Verification")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",    str(a.get("accuracy", "?")))
        c2.metric("Macro F1",    str(a.get("macro_f1", "?")))
        c3.metric("Exact Match", str(a.get("exact_match", "?")))
        c4.metric("BLEU",        str(a.get("bleu", "?")))
        c5.metric("ROUGE-L",     str(a.get("rouge_l", "?")))

        c6, _ = st.columns(2)
        c6.metric("METEOR", str(a.get("meteor", "?")))
        st.caption(f"Classifier used: {a.get('model_used', '?')}")

        if "confusion_matrix" in a:
            try:
                import plotly.figure_factory as ff
                fig = ff.create_annotated_heatmap(
                    a["confusion_matrix"],
                    x=["Pred: incorrect", "Pred: correct"],
                    y=["True: incorrect", "True: correct"],
                    colorscale="Blues",
                )
                fig.update_layout(title="Confusion Matrix — Model A (Test Set)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.json(a["confusion_matrix"])

        st.divider()

        # ── Unsupervised ───────────────────────────────────────────────────────
        st.subheader("Unsupervised Models")
        u1, u2, u3 = st.columns(3)
        u1.metric("KMeans Silhouette", str(u.get("kmeans_silhouette", "?")))
        u2.metric("KMeans Purity",     str(u.get("kmeans_purity", "?")))
        u3.metric("LabelProp Macro F1",str(u.get("label_prop_f1", "?")))

        st.divider()

        # ── Model B ────────────────────────────────────────────────────────────
        st.subheader("Model B — Distractor & Hint Generation")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Dist BLEU",    str(b.get("distractor_bleu", "?")))
        d2.metric("Dist ROUGE-L", str(b.get("distractor_rouge_l", "?")))
        d3.metric("Dist METEOR",  str(b.get("distractor_meteor", "?")))
        d4.metric("Dist Accuracy",str(b.get("distractor_accuracy", "?")))

        h1, h2 = st.columns(2)
        h1.metric("Hint-3 ROUGE-L", str(b.get("hint_rouge_l", "?")))
        h2.metric("Hint-3 METEOR",  str(b.get("hint_meteor", "?")))
        st.caption(f"Evaluated on {b.get('n_samples_evaluated', '?')} test samples.")

        st.divider()

        # ── Benchmark comparison ───────────────────────────────────────────────
        st.subheader("Benchmark Comparison (BERT / T5 — reference only, not trained here)")
        bench_df = pd.DataFrame([
            {"Model": "BERT-base (fine-tuned)",   "Accuracy": f"{ref.get('bert_base_accuracy', 0):.1%}",  "Source": ref.get("source", "")},
            {"Model": "BERT-large (fine-tuned)",  "Accuracy": f"{ref.get('bert_large_accuracy', 0):.1%}", "Source": ref.get("source", "")},
            {"Model": "T5-base (fine-tuned)",      "Accuracy": f"{ref.get('t5_base_accuracy', 0):.1%}",   "Source": ref.get("source", "")},
            {"Model": f"Your model ({a.get('model_used','?')})", "Accuracy": str(a.get("exact_match", "?")), "Source": "This project"},
        ])
        st.table(bench_df)
        st.caption(
            "Classical ML on RACE typically achieves 30–50 % Exact Match. "
            "The gap vs BERT/T5 is expected — document it in your report as motivation "
            "for neural approaches."
        )

    st.divider()

    # ── Session log ────────────────────────────────────────────────────────────
    st.subheader("Session Log")
    log = st.session_state.get("session_log", [])
    if log:
        log_df    = pd.DataFrame(log)
        st.dataframe(log_df, use_container_width=True)
        n_correct = int(log_df["correct"].sum())
        st.metric("Your score this session",
                  f"{n_correct}/{len(log_df)}  ({n_correct/len(log_df):.0%})")
        st.download_button(
            label="📥 Download Session Log",
            data=log_df.to_csv(index=False),
            file_name="session_log.csv",
            mime="text/csv",
        )
    else:
        st.info("No questions answered yet this session.  Try the Quiz screen!")
