"""
Semantic retrieval module for extracting key sentences.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .utils import simple_sent_tokenize


def get_device() -> torch.device:
    """Get the best available device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_key_sentences(
    docs: list[str],
    query: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 3,
    device: torch.device = None
) -> tuple[list[str], list[float]]:
    """
    Extract top-k key sentences from documents based on semantic similarity to query.

    Args:
        docs: List of document passages
        query: User query
        model_name: SentenceTransformer model name
        top_k: Number of top sentences to return
        device: Torch device (auto-detected if None)

    Returns:
        Tuple of (sentences, scores)
    """
    if device is None:
        device = get_device()

    # Tokenize documents into sentences
    sentences = []
    for doc in docs:
        sentences.extend(simple_sent_tokenize(doc))

    if not sentences:
        return [], []

    # Load embedding model
    embedder = SentenceTransformer(model_name, device=str(device))

    # Encode sentences and query
    sent_embs = embedder.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    query_emb = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # Compute cosine similarity
    scores = np.dot(sent_embs, query_emb)

    # Get top-k indices
    top_k = min(top_k, len(sentences))
    top_idx = np.argsort(scores)[-top_k:][::-1]

    return (
        [sentences[i] for i in top_idx],
        [float(scores[i]) for i in top_idx]
    )
