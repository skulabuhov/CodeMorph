import json
import os
from typing import List, Optional

import faiss
import numpy as np
from openai import AsyncOpenAI


class FaissVectorStore:
    """Vector store backed by FAISS and OpenAI embeddings."""

    def __init__(self, client: AsyncOpenAI, max_size: int = 300) -> None:
        self.client = client
        self.max_size = max_size
        self.embeddings: List[List[float]] = []
        self.texts: List[str] = []
        self.index: Optional[faiss.IndexFlatL2] = None

    async def add(self, text: str) -> None:
        embedding = await self._embed(text)
        self.embeddings.append(embedding)
        self.texts.append(text)
        if len(self.texts) > self.max_size:
            self.embeddings.pop(0)
            self.texts.pop(0)
        self._rebuild_index()

    async def search(self, query: str, k: int = 3) -> List[str]:
        if not self.texts:
            return []
        q_emb = await self._embed(query)
        self._ensure_index()
        k = min(k, len(self.texts))
        D, I = self.index.search(np.array([q_emb], dtype="float32"), k)
        return [self.texts[i] for i in I[0] if i >= 0]

    async def _embed(self, text: str) -> List[float]:
        resp = await self.client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return resp.data[0].embedding

    def _rebuild_index(self) -> None:
        if not self.embeddings:
            self.index = None
            return
        dim = len(self.embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        emb = np.array(self.embeddings, dtype="float32")
        self.index.add(emb)

    def _ensure_index(self) -> None:
        if self.index is None:
            self._rebuild_index()

    def save(self, path: str) -> None:
        self._ensure_index()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path + ".index")
        with open(path + ".json", "w") as f:
            json.dump(self.texts, f)

    @classmethod
    def load(cls, client: AsyncOpenAI, path: str) -> "FaissVectorStore":
        store = cls(client)
        index_file = path + ".index"
        texts_file = path + ".json"
        if os.path.exists(index_file) and os.path.exists(texts_file):
            store.index = faiss.read_index(index_file)
            with open(texts_file) as f:
                store.texts = json.load(f)
            if store.index is not None:
                n = store.index.ntotal
                if n:
                    vecs = store.index.reconstruct_n(0, n)
                    store.embeddings = vecs.tolist()
        else:
            # If the index or text files are missing, an empty store is returned
            pass
        return store
