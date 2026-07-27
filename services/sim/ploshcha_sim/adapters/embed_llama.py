from ..ports.embedding import EmbeddingPort


class LlamaEmbedder(EmbeddingPort):
    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1", model: str = "bge-m3",
                 dim: int = 1024, timeout: float = 180.0):
        from openai import OpenAI
        self.model = model
        self.dim = dim
        self.calls = 0
        self._client = OpenAI(base_url=base_url, api_key="dummy", timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        r = self._client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in r.data]
