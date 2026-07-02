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
        (
            "avoid candidates who only have generic consulting background, pure tutorials, unrelated computer vision, "
            "speech, design, marketing, HR, sales, or keyword stuffing without production AI evidence"
        ),
    ]


def compute_semantic_scores(
    df,
    jd_text,
    model_name=DEFAULT_MODEL,
    device="auto",
    batch_size=256,
    top_k=25000,
):
    device = get_device(device)

    print(f"Embedding model: {model_name}")
    print(f"Embedding device: {device}")

    model = SentenceTransformer(model_name, device=device)

    jd_queries = build_jd_queries(jd_text)

    query_embeddings = model.encode(
        jd_queries,
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    candidate_texts = df["semantic_text"].tolist()

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

    top_k = min(top_k, len(df))

    distances, indices = index.search(query_embeddings, top_k)

    max_semantic = np.zeros(len(df), dtype=np.float32)
    avg_semantic = np.zeros(len(df), dtype=np.float32)
    counts = np.zeros(len(df), dtype=np.float32)

    for query_idx in range(len(jd_queries)):
        idxs = indices[query_idx]
        vals = distances[query_idx]

        max_semantic[idxs] = np.maximum(max_semantic[idxs], vals)
        avg_semantic[idxs] += vals
        counts[idxs] += 1

    avg_semantic = np.divide(
        avg_semantic,
        counts,
        out=np.zeros_like(avg_semantic),
        where=counts > 0,
    )

    semantic_score = 0.75 * max_semantic + 0.25 * avg_semantic

    return semantic_score