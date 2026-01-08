import os
import pickle
from typing import List, Dict

from rank_bm25 import BM25Okapi
import jieba

from config.settings import settings


class SparseRetriever:
    def __init__(self, lazy: bool = False):
        self.bm25: BM25Okapi | None = None
        self.corpus: List[str] = []
        if not lazy:
            self._load_index()

    def _load_index(self):
        path = settings.bm25_index_path
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.corpus = data["corpus"]
                tokenized = [self._tokenize(doc) for doc in self.corpus]
                self.bm25 = BM25Okapi(tokenized)

    def _tokenize(self, text: str) -> List[str]:
        return list(jieba.cut(text))

    def build_index(self, texts: List[str]):
        self.corpus = texts
        tokenized = [self._tokenize(doc) for doc in texts]
        self.bm25 = BM25Okapi(tokenized)
        os.makedirs(os.path.dirname(settings.bm25_index_path) or ".", exist_ok=True)
        with open(settings.bm25_index_path, "wb") as f:
            pickle.dump({"corpus": self.corpus}, f)

    def bm25_search(self, query: str, top_k: int = 20) -> List[Dict]:
        if self.bm25 is None:
            return []
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        hits = []
        for idx in top_indices:
            if scores[idx] > 0:
                hits.append({
                    "text": self.corpus[idx],
                    "metadata": {},
                    "sparse_score": float(scores[idx]),
                })
        return hits

    def count(self) -> int:
        return len(self.corpus)


sparse_retriever = SparseRetriever()
