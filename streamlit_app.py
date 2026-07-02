import ssl
from pathlib import Path

import pandas as pd
import streamlit as st

_original_load_default_certs = ssl.SSLContext.load_default_certs

def _safe_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        return _original_load_default_certs(self, purpose)
    except ssl.SSLError:
        try:
            self.set_default_verify_paths()
        except Exception:
            pass

ssl.SSLContext.load_default_certs = _safe_load_default_certs


ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "data" / "output" / "submission.csv"
DEBUG_PATH = ROOT / "data" / "output" / "top100_debug.csv"


st.set_page_config(
    page_title="FitRank AI Sandbox",
    page_icon="🎯",
    layout="wide",
)


st.title("🎯 FitRank AI Sandbox")
st.caption("Hybrid AI candidate ranking system for Redrob x H2S INDIA.RUNS Data & AI Challenge")


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def load_data():
    submission = None
    debug = None

    if OUT_PATH.exists():
        submission = load_csv(OUT_PATH)

    if DEBUG_PATH.exists():
        debug = load_csv(DEBUG_PATH)

    return submission, debug


submission, debug = load_data()


with st.sidebar:
    st.header("Sandbox Controls")

    uploaded_submission = st.file_uploader(
        "Upload submission.csv",
        type=["csv"],
    )

    uploaded_debug = st.file_uploader(
        "Upload top100_debug.csv",
        type=["csv"],
    )

    if uploaded_submission is not None:
        submission = pd.read_csv(uploaded_submission)

    if uploaded_debug is not None:
        debug = pd.read_csv(uploaded_debug)

    st.divider()

    st.subheader("Pipeline Summary")
    st.write("Model: `all-MiniLM-L6-v2`")
    st.write("Vector Search: `FAISS`")
    st.write("Ranking Type: `Hybrid`")
    st.write("Runtime: `~3 mins`")
    st.write("Candidates Processed: `100,000`")
    st.write("Output: `Top 100`")


if submission is None:
    st.warning(
        "No submission file found. Run the ranking pipeline first or upload submission.csv from the sidebar."
    )
    st.code(
        "python rank.py --raw_dir data/raw --out data/output/submission.csv --device cuda --batch_size 256 --top_k 25000 --candidate_pool 30000",
        language="bash",
    )
    st.stop()


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🏆 Leaderboard",
        "🔍 Candidate Explainer",
        "📊 Scoring Breakdown",
        "🧠 Methodology",
        "✅ Validation",
    ]
)


with tab1:
    st.subheader("Top Ranked Candidates")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Output Rows", len(submission))
    c2.metric("Unique Candidates", submission["candidate_id"].nunique())
    c3.metric("Top Score", round(float(submission["score"].max()), 4))
    c4.metric("Lowest Top-100 Score", round(float(submission["score"].min()), 4))

    top_n = st.slider("Show top N candidates", 5, 100, 20)

    if debug is not None:
        show_cols = [
            col for col in [
                "rank",
                "candidate_id",
                "title",
                "years_experience",
                "location",
                "score",
                "semantic_score",
                "lexical_score",
                "ir_nlp_score",
                "skill_depth_score",
                "production_score",
                "behavior_score",
                "honeypot_penalty",
                "reasoning",
            ]
            if col in debug.columns
        ]
        st.dataframe(debug[show_cols].head(top_n), use_container_width=True)
    else:
        st.dataframe(submission.head(top_n), use_container_width=True)


with tab2:
    st.subheader("Candidate-Level Explanation")

    source_df = debug if debug is not None else submission

    candidate_options = source_df["candidate_id"].astype(str).tolist()

    selected_candidate = st.selectbox(
        "Select candidate",
        candidate_options,
    )

    row = source_df[source_df["candidate_id"].astype(str) == selected_candidate].iloc[0]

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### Candidate Summary")

        st.write(f"**Candidate ID:** `{row.get('candidate_id', 'N/A')}`")
        st.write(f"**Rank:** `{row.get('rank', 'N/A')}`")
        st.write(f"**Score:** `{round(float(row.get('score', 0)), 6)}`")

        if "title" in row:
            st.write(f"**Title:** {row.get('title', 'N/A')}")

        if "years_experience" in row:
            st.write(f"**Experience:** {row.get('years_experience', 'N/A')} years")

        if "location" in row:
            st.write(f"**Location:** {row.get('location', 'N/A')}")

        if "reasoning" in row:
            st.markdown("### Explanation")
            st.info(row.get("reasoning", ""))

    with right:
        st.markdown("### Score Components")

        score_cols = [
            "semantic_score",
            "lexical_score",
            "ir_nlp_score",
            "skill_depth_score",
            "production_score",
            "title_score",
            "experience_score",
            "company_score",
            "behavior_score",
            "honeypot_penalty",
        ]

        available_score_cols = [col for col in score_cols if col in source_df.columns]

        if available_score_cols:
            component_df = pd.DataFrame(
                {
                    "signal": available_score_cols,
                    "value": [float(row[col]) for col in available_score_cols],
                }
            )

            st.bar_chart(
                component_df.set_index("signal"),
                height=350,
            )
        else:
            st.write("Debug score columns not available. Upload top100_debug.csv for full explanation.")


with tab3:
    st.subheader("Ranking Signal Analysis")

    if debug is None:
        st.warning("Upload or generate top100_debug.csv to see scoring analysis.")
    else:
        numeric_cols = [
            col for col in [
                "semantic_score",
                "lexical_score",
                "ir_nlp_score",
                "skill_depth_score",
                "production_score",
                "title_score",
                "experience_score",
                "company_score",
                "behavior_score",
                "honeypot_penalty",
                "score",
            ]
            if col in debug.columns
        ]

        if numeric_cols:
            st.markdown("### Average Signal Strength in Top 100")
            avg_df = (
                debug[numeric_cols]
                .mean()
                .reset_index()
                .rename(columns={"index": "signal", 0: "average_value"})
            )

            st.bar_chart(avg_df.set_index("signal"), height=350)

            st.markdown("### Top 20 Score Trend")
            trend_cols = [col for col in ["rank", "score"] if col in debug.columns]
            if len(trend_cols) == 2:
                trend_df = debug[trend_cols].head(20).set_index("rank")
                st.line_chart(trend_df, height=300)

        st.markdown("### Raw Debug Table")
        st.dataframe(debug.head(100), use_container_width=True)


with tab4:
    st.subheader("How FitRank AI Works")

    st.markdown(
        """
        FitRank AI is a hybrid candidate ranking engine designed to go beyond keyword matching.

        **1. JD Understanding**  
        The job description is converted into multiple recruiter-style search intents:
        semantic search, ranking systems, AI engineering, NLP, recommendation systems,
        vector databases, FAISS, production ML, and startup ownership.

        **2. Candidate Representation**  
        Each candidate profile is converted into a structured text representation using:
        title, headline, summary, skills, career history, education, certifications,
        location, industry, and experience.

        **3. Fast Pre-Filtering**  
        All 100,000 candidates are scored using TF-IDF and structured signals.
        The strongest 30,000 are selected for semantic ranking.

        **4. Semantic Retrieval**  
        SentenceTransformer embeddings are generated for the selected candidate pool.
        FAISS inner-product search retrieves candidates most semantically aligned with the JD.

        **5. Final Ranking**  
        Multiple signals are combined:
        semantic similarity, lexical similarity, skill depth, IR/NLP evidence,
        production readiness, title fit, experience fit, company relevance,
        behavioral signals, and suspicious-profile penalties.

        **6. Explainability**  
        Every top-100 candidate receives a short explanation generated only from
        available profile evidence and score signals.
        """
    )

    st.markdown("### System Pipeline")

    st.code(
        """
candidates.jsonl
      ↓
Profile Parsing + Text Normalization
      ↓
TF-IDF + Structured Feature Scoring on 100K
      ↓
Candidate Pool Selection: Top 30K
      ↓
SentenceTransformer Embeddings
      ↓
FAISS Semantic Retrieval
      ↓
Hybrid Weighted Ranking
      ↓
Honeypot / Suspicious Profile Penalty
      ↓
Top 100 Submission + Explanations
        """,
        language="text",
    )


with tab5:
    st.subheader("Submission Validation Checks")

    checks = {
        "Exactly 100 rows": len(submission) == 100,
        "Unique candidate IDs": submission["candidate_id"].nunique() == len(submission),
        "Ranks are 1 to 100": submission["rank"].tolist() == list(range(1, 101)),
        "Scores are non-increasing": submission["score"].is_monotonic_decreasing,
        "No missing candidate IDs": submission["candidate_id"].notna().all(),
        "No missing scores": submission["score"].notna().all(),
        "No missing reasoning": submission["reasoning"].notna().all()
        if "reasoning" in submission.columns
        else False,
    }

    for check, passed in checks.items():
        if passed:
            st.success(f"✅ {check}")
        else:
            st.error(f"❌ {check}")

    st.markdown("### Submission Preview")
    st.dataframe(submission.head(100), use_container_width=True)

    csv_bytes = submission.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download submission.csv",
        data=csv_bytes,
        file_name="submission.csv",
        mime="text/csv",
    )