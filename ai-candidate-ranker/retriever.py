"""
retriever.py — Step 3: First-Pass Retrieval (Semantic + Hybrid Search).

Performs fast candidate filtering via FAISS cosine similarity,
optional BM25 keyword scoring, and weighted hybrid fusion.
"""

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
#  Semantic Search
# ======================================================================

def semantic_search(
    index,
    query_vector: np.ndarray,
    top_n: int = 50,
) -> list[tuple[int, float]]:
    """
    Search a FAISS index for the top-N candidates by cosine similarity.

    Parameters
    ----------
    index : faiss.Index
        Pre-built FAISS index of candidate vectors.
    query_vector : np.ndarray
        The JD embedding vector, shape (dim,) or (1, dim).
    top_n : int
        Number of candidates to retrieve.

    Returns
    -------
    list of (candidate_index, score) sorted descending by score.
    """
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    # Clamp top_n to available vectors
    top_n = min(top_n, index.ntotal)

    scores, indices = index.search(query_vector, top_n)
    results = [
        (int(idx), float(score))
        for idx, score in zip(indices[0], scores[0])
        if idx >= 0  # FAISS returns -1 for empty slots
    ]

    logger.info("Semantic search returned %d results.", len(results))
    return results


# ======================================================================
#  BM25 Keyword Search
# ======================================================================

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9]+(?:['\-][a-z0-9]+)*\b", text)
    return tokens


def bm25_search(
    corpus: list[str],
    query: str,
    top_n: int = 50,
) -> list[tuple[int, float]]:
    """
    Score candidates against the JD using BM25 (Okapi).

    Parameters
    ----------
    corpus : list[str]
        Candidate documents (same order as the DataFrame).
    query : str
        The full JD text.
    top_n : int
        Number of results to return.

    Returns
    -------
    list of (candidate_index, score) sorted descending.
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_tokenize(doc) for doc in corpus]
    tokenized_query = _tokenize(query)

    bm25 = BM25Okapi(tokenized_corpus)
    raw_scores = bm25.get_scores(tokenized_query)

    # Pair with indices and sort
    scored = [(i, float(s)) for i, s in enumerate(raw_scores)]
    scored.sort(key=lambda x: x[1], reverse=True)

    results = scored[:top_n]
    logger.info("BM25 search returned %d results.", len(results))
    return results


# ======================================================================
#  Min-Max Normalisation
# ======================================================================

def _normalize_scores(results: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Normalise scores to [0, 1] via min-max scaling."""
    if not results:
        return results

    scores = [s for _, s in results]
    lo, hi = min(scores), max(scores)
    rng = hi - lo if hi != lo else 1.0

    return [(idx, (s - lo) / rng) for idx, s in results]


# ======================================================================
#  Hybrid Search (Semantic + BM25)
# ======================================================================

def hybrid_search(
    semantic_results: list[tuple[int, float]],
    bm25_results: list[tuple[int, float]],
    alpha: float = 0.70,
    top_n: Optional[int] = None,
) -> list[tuple[int, float]]:
    """
    Combine semantic and BM25 scores via weighted linear fusion.

    final_score = α * semantic_score_norm + (1 - α) * bm25_score_norm

    Parameters
    ----------
    semantic_results : list of (index, score)
    bm25_results : list of (index, score)
    alpha : float
        Weight for semantic component. (1 - alpha) goes to BM25.
    top_n : int, optional
        If set, truncate the fused results to top_n.

    Returns
    -------
    list of (candidate_index, fused_score) sorted descending.
    """
    # Normalise both score lists
    sem_norm = dict(_normalize_scores(semantic_results))
    bm25_norm = dict(_normalize_scores(bm25_results))

    # Union of all candidate indices
    all_indices = set(sem_norm.keys()) | set(bm25_norm.keys())

    fused = []
    for idx in all_indices:
        s_sem = sem_norm.get(idx, 0.0)
        s_bm25 = bm25_norm.get(idx, 0.0)
        combined = alpha * s_sem + (1 - alpha) * s_bm25
        fused.append((idx, combined))

    fused.sort(key=lambda x: x[1], reverse=True)

    if top_n is not None:
        fused = fused[:top_n]

    logger.info(
        "Hybrid search (α=%.2f) produced %d fused results.",
        alpha,
        len(fused),
    )
    return fused
