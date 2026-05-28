"""
embedder.py — Step 2: Semantic Representation (Vector Embeddings).

Loads a sentence-transformer model, encodes text documents into
dense vectors, and builds a FAISS index for fast similarity search.
"""

import logging
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Wraps a sentence-transformer model and a FAISS index."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model identifier for sentence-transformers.
        """
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.dimension: int = self.model.get_sentence_embedding_dimension()
        self.index: Optional[faiss.Index] = None
        logger.info(
            "Model loaded. Embedding dimension: %d", self.dimension
        )

    # ──────────────────────────────────────────────────────────────────
    #  Encoding
    # ──────────────────────────────────────────────────────────────────

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of text strings into dense vectors.

        Parameters
        ----------
        texts : list[str]
            Documents to encode.
        batch_size : int
            Encoding batch size (tune for GPU memory).
        show_progress : bool
            Show a progress bar during encoding.
        normalize : bool
            L2-normalize vectors (required for cosine similarity via
            inner-product index).

        Returns
        -------
        np.ndarray of shape (len(texts), dimension), dtype float32
        """
        logger.info("Encoding %d documents …", len(texts))
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        logger.info("Encoding complete. Shape: %s", vectors.shape)
        return vectors

    # ──────────────────────────────────────────────────────────────────
    #  FAISS Index
    # ──────────────────────────────────────────────────────────────────

    def build_index(self, vectors: np.ndarray) -> faiss.Index:
        """
        Build a FAISS inner-product (cosine) index from candidate vectors.

        Because vectors are L2-normalised, inner product == cosine similarity.

        Parameters
        ----------
        vectors : np.ndarray
            Matrix of shape (n_candidates, dimension).

        Returns
        -------
        faiss.IndexFlatIP
        """
        n, d = vectors.shape
        assert d == self.dimension, (
            f"Vector dimension {d} != model dimension {self.dimension}"
        )

        logger.info("Building FAISS IndexFlatIP with %d vectors …", n)
        index = faiss.IndexFlatIP(d)
        index.add(vectors)
        self.index = index
        logger.info("FAISS index ready. Total vectors: %d", index.ntotal)
        return index

    def search(
        self,
        query_vector: np.ndarray,
        top_n: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search the FAISS index for the top-N nearest candidates.

        Parameters
        ----------
        query_vector : np.ndarray
            Shape (1, dimension) or (dimension,).
        top_n : int
            Number of results to return.

        Returns
        -------
        scores : np.ndarray of shape (top_n,)
            Cosine similarity scores (higher = more similar).
        indices : np.ndarray of shape (top_n,)
            Row indices into the original candidate matrix.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        scores, indices = self.index.search(query_vector, top_n)
        return scores[0], indices[0]
