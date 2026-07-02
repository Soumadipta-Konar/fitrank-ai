import ssl

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


import argparse
from pathlib import Path

from src.data_loader import get_bundle_paths, read_docx, load_candidates
from src.semantic_ranker import compute_semantic_scores
from src.lexical_ranker import compute_lexical_scores
from src.features import add_feature_scores
from src.prefilter import select_candidate_pool
from src.scoring import add_final_scores
from src.submission import create_submission, validate_submission


def run_pipeline(
    raw_dir,
    out_path,
    debug_path,
    model_name,
    device,
    batch_size,
    top_k,
    candidate_pool,
):
    raw_dir = Path(raw_dir)
    out_path = Path(out_path)
    debug_path = Path(debug_path) if debug_path else None

    print("=" * 80)
    print("FitRank AI: Optimized Hybrid Semantic Candidate Ranking")
    print("=" * 80)

    print("\n[1/9] Finding competition files...")
    paths = get_bundle_paths(raw_dir)

    print("Candidates:", paths["candidates"])
    print("Job description:", paths["job_description"])
    print("Validator:", paths["validator"])

    print("\n[2/9] Reading job description...")
    jd_text = read_docx(paths["job_description"])
    print("JD characters:", len(jd_text))

    print("\n[3/9] Loading candidate profiles...")
    df = load_candidates(paths["candidates"])
    print("Loaded candidates:", len(df))

    print("\n[4/9] Computing TF-IDF lexical scores on all candidates...")
    df["lexical_score"] = compute_lexical_scores(df)

    print("\n[5/9] Computing structured feature scores on all candidates...")
    df = add_feature_scores(df)

    print("\n[6/9] Selecting candidate pool for semantic ranking...")
    pool_indices = select_candidate_pool(df, pool_size=candidate_pool)
    print("Selected candidate pool:", len(pool_indices))

    print("\n[7/9] Computing semantic scores using SentenceTransformer + FAISS on pool...")
    df["semantic_score"] = compute_semantic_scores(
        df=df,
        jd_text=jd_text,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        top_k=top_k,
        candidate_indices=pool_indices,
    )

    print("\n[8/9] Computing final ranking...")
    df = add_final_scores(df)

    print("\nTop 10 candidates:")
    preview_cols = [
        "candidate_id",
        "title",
        "years_experience",
        "location",
        "final_score_raw",
        "semantic_score",
        "lexical_score",
        "ir_nlp_score",
        "skill_depth_score",
        "behavior_score",
        "honeypot_penalty",
    ]
    print(df[preview_cols].head(10).to_string(index=False))

    print("\n[9/9] Creating submission...")
    submission = create_submission(
        df=df,
        out_path=out_path,
        debug_path=debug_path,
    )

    print("Saved submission:", out_path)
    if debug_path:
        print("Saved debug file:", debug_path)

    print("\nValidating submission format...")
    stdout, stderr = validate_submission(paths["validator"], out_path)

    print("\nValidator STDOUT:")
    print(stdout)

    if stderr.strip():
        print("\nValidator STDERR:")
        print(stderr)

    print("\nDone.")
    return submission


def parse_args():
    parser = argparse.ArgumentParser(
        description="FitRank AI optimized candidate ranking pipeline"
    )

    parser.add_argument(
        "--raw_dir",
        default="data/raw",
        help="Directory containing competition ZIP",
    )

    parser.add_argument(
        "--out",
        default="data/output/submission.csv",
        help="Output submission CSV path",
    )

    parser.add_argument(
        "--debug",
        default="data/output/top100_debug.csv",
        help="Debug CSV path",
    )

    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Embedding device",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Embedding batch size",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=25000,
        help="FAISS retrieval pool size per JD query",
    )

    parser.add_argument(
        "--candidate_pool",
        type=int,
        default=30000,
        help="Number of candidates selected for semantic embedding",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_pipeline(
        raw_dir=args.raw_dir,
        out_path=args.out,
        debug_path=args.debug,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        top_k=args.top_k,
        candidate_pool=args.candidate_pool,
    )