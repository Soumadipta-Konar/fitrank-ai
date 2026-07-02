import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_device(device="auto"):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_jd_queries(jd_text):
    return [
        jd_text,
        (
            "Senior AI Engineer founding team embeddings retrieval ranking semantic search "
            "vector search LLM fine tuning NLP Python production systems evaluation NDCG MRR MAP"
        ),
        (
            "AI engineer who shipped production search ranking recommendation retrieval matching systems "
            "with real users latency constraints monitoring and evaluation"
        ),
        (
            "candidate job matching recruiter ranking hybrid search vector database FAISS Milvus Pinecone "
            "Qdrant Weaviate Elasticsearch OpenSearch semantic embeddings"
        ),
        (
            "strong product engineer startup scrappy ownership backend Python ML systems deployed production code"
        ),
    ]


def compute_semantic_scores(
    df,
    jd_text,
    model_name=DEFAULT_MODEL,
    device="auto",
    batch_size=256,
    top_k=25000,
    candidate_indices=None,
):
    device = get_device(device)

    print(f"Embedding model: {model_name}")
    print(f"Embedding device: {device}")

    if candidate_indices is None:
        candidate_indices = np.arange(len(df))
    else:
        candidate_indices = np.asarray(candidate_indices)

    pool_df = df.iloc[candidate_indices].copy()

    print(f"Semantic pool size: {len(pool_df)} / {len(df)}")

    model = SentenceTransformer(model_name, device=device)

    jd_queries = build_jd_queries(jd_text)

    query_embeddings = model.encode(
        jd_queries,
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    candidate_texts = pool_df["semantic_text"].tolist()

    candidate_embeddings = model.encode(
        candidate_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    dim = candidate_embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(candidate_embeddings)

    top_k = min(top_k, len(pool_df))

    distances, local_indices = index.search(query_embeddings, top_k)

    pool_scores_max = np.zeros(len(pool_df), dtype=np.float32)
    pool_scores_avg = np.zeros(len(pool_df), dtype=np.float32)
    counts = np.zeros(len(pool_df), dtype=np.float32)

    for query_idx in range(len(jd_queries)):
        idxs = local_indices[query_idx]
        vals = distances[query_idx]

        pool_scores_max[idxs] = np.maximum(pool_scores_max[idxs], vals)
        pool_scores_avg[idxs] += vals
        counts[idxs] += 1

    pool_scores_avg = np.divide(
        pool_scores_avg,
        counts,
        out=np.zeros_like(pool_scores_avg),
        where=counts > 0,
    )

    pool_semantic_score = 0.75 * pool_scores_max + 0.25 * pool_scores_avg

    full_scores = np.zeros(len(df), dtype=np.float32)
    full_scores[candidate_indices] = pool_semantic_score

    return full_scores