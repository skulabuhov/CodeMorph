import math
from typing import List

from openai import AsyncOpenAI


class SimpleVectorStore:
    """Very small in-memory vector store using OpenAI embeddings."""

    def __init__(self, client: AsyncOpenAI) -> None:
        self.client = client
        self.embeddings: List[List[float]] = []
        self.texts: List[str] = []

    async def add(self, text: str) -> None:
        embedding = await self._embed(text)
        self.embeddings.append(embedding)
        self.texts.append(text)

    async def search(self, query: str, k: int = 3) -> List[str]:
        if not self.embeddings:
            return []
        q_emb = await self._embed(query)
        scores = [self._cosine_similarity(q_emb, emb) for emb in self.embeddings]
        pairs = sorted(zip(scores, self.texts), reverse=True)
        return [t for _, t in pairs[:k]]

    async def _embed(self, text: str) -> List[float]:
        resp = await self.client.embeddings.create(model="text-embedding-3-small", input=text)
        return resp.data[0].embedding

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
