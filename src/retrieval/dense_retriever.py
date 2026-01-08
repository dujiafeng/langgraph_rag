import os
from typing import List, Dict

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from config.settings import settings
from src.utils.embeddings import get_embeddings


class DenseRetriever:
    def __init__(self):
        persist_dir = os.path.abspath(settings.vector_store_path)
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(allow_reset=True, anonymized_telemetry=False),
        )
        self.collection_name = "rag_docs"
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except (ValueError, NotFoundError):
            self.collection = self.client.create_collection(self.collection_name)

    def add_documents(self, texts: List[str], metadatas: List[Dict] | None = None, ids: List[str] | None = None):
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        embeddings = get_embeddings().embed_documents(texts)
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas or [{}] * len(texts),
            ids=ids,
        )

    def similarity_search(self, query: str, top_k: int = 20) -> List[Dict]:
        query_emb = get_embeddings().embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, 100),
        )
        hits = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                hits.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "dense_score": 1.0 - results["distances"][0][i] if results.get("distances") else 0.0,
                })
        return hits

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()


dense_retriever = DenseRetriever()
