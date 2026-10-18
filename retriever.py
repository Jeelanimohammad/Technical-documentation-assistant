"""
retriever.py
============
Vector embedding generation and semantic similarity search using
Sentence Transformers and FAISS (Facebook AI Similarity Search).
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class DocumentRetriever:
    """Manages text embeddings and similarity search using FAISS."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the DocumentRetriever with an open-source embedding model.

        'all-MiniLM-L6-v2' is a lightweight (approx 80MB), fast, and state-of-the-art
        compact transformer model producing 384-dimensional dense embeddings.
        It runs efficiently on standard laptop CPUs without GPU requirements.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        try:
            self.dimension = self.model.get_embedding_dimension()
        except AttributeError:
            self.dimension = self.model.get_sentence_embedding_dimension()
        self.index: faiss.IndexFlatIP = None
        self.chunks_metadata: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Compute normalized dense embeddings for chunks and index them in FAISS.

        We use IndexFlatIP (Inner Product) on L2-normalized vectors,
        which computes exact Cosine Similarity.

        Args:
            chunks (List[Dict[str, Any]]): List of chunk metadata dictionaries.

        Returns:
            int: Number of chunks indexed.
        """
        if not chunks:
            self.index = None
            self.chunks_metadata = []
            return 0

        self.chunks_metadata = chunks
        texts = [chunk["text"] for chunk in chunks]

        # Generate embeddings as float32 numpy array
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalizes vectors to unit length for Cosine Similarity
        ).astype(np.float32)

        # Create FAISS Flat Inner Product index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        return len(chunks)

    def retrieve(self, query: str, top_k: int = 3, min_similarity_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most semantically relevant chunks for a user query.

        Args:
            query (str): User's natural language question.
            top_k (int): Number of top passages to retrieve.
            min_similarity_threshold (float): Minimum cosine similarity score.

        Returns:
            List[Dict[str, Any]]: Relevant chunks with similarity scores.
        """
        if not query.strip() or self.index is None or len(self.chunks_metadata) == 0:
            return []

        # Encode user question and normalize
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        k = min(top_k, len(self.chunks_metadata))
        similarities, indices = self.index.search(query_vector, k)

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            # FAISS returns -1 for empty slots
            if idx == -1:
                continue

            similarity_score = float(sim)
            # Filter out chunks that do not meet minimum relevance threshold
            if similarity_score >= min_similarity_threshold:
                chunk_copy = dict(self.chunks_metadata[idx])
                chunk_copy["similarity_score"] = round(similarity_score, 4)
                results.append(chunk_copy)

        return results
