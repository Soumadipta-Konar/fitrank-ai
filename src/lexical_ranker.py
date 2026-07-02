import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


DEFAULT_QUERY = """
Senior AI Engineer founding team production AI engineer machine learning engineer
embeddings semantic search retrieval ranking recommendation recommender search
vector database FAISS Milvus Pinecone Qdrant Weaviate Elasticsearch OpenSearch
LLM fine-tuning LoRA NLP Python PyTorch transformers evaluation NDCG MRR MAP
A/B testing offline evaluation matching candidates jobs recruiter marketplace
production systems backend latency monitoring deployed shipped ownership startup
"""


def minmax(values):
    values = np.asarray(values, dtype=float)
    return (values - values.min()) / (values.max() - values.min() + 1e-9)


def compute_lexical_scores(df, query_text=DEFAULT_QUERY):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=80000,
    )

    candidate_matrix = vectorizer.fit_transform(df["semantic_text"])
    query_vector = vectorizer.transform([query_text])

    scores = linear_kernel(query_vector, candidate_matrix).flatten()

    return minmax(scores)