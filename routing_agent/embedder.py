from sentence_transformer import SentenceTransformer

class Embedder:
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(self, text: str) -> list[float]:
        embeddings = self.model.encode(text, normalize_embedings=True)
        return embeddings.tolist()

        