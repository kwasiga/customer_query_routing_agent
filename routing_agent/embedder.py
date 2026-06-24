# embedder.py
#
# Wraps the sentence-transformers embedding model.
# Converts raw text (queries or documents) into fixed-size float vectors
# that can be stored in or searched against VectorAI.
# Contains:
#   - Embedder class
#       - embed()         : embeds a single string into a vector
#       - _embed_batch()  : embeds a list of strings in one pass (used during ingestion)

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from routing_agent.config import EMBEDDING_MODEL


class Embedder:
    def __init__(self) -> None:
        # Loads the embedding model from HuggingFace on first use
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding model ready!")

    def embed(self, text: str) -> list[float]:
        # Converts a single string into a normalised float vector
        vector = self.model.encode(text, show_progress_bar=False, normalize_embeddings=True)
        return vector.tolist()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Converts a list of strings into vectors in a single model pass — more efficient than calling embed() in a loop
        vectors = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return [v.tolist() for v in vectors]
