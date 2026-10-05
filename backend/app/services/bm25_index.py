"""
BM25 keyword index over the same chunk list used by the FAISS vector
store (each chunk: {"text": ..., "metadata": {...}}).

This complements semantic FAISS search: BM25 catches exact term
matches (ticker symbols, GAAP line-item names, exact dollar amounts)
that dense embeddings often blur past.

Chunks are referenced by their position in the shared `all_chunks`
list built in main.py, so BM25 results and FAISS results can be
fused (RRF) just by matching that index.
"""

import re
from typing import Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return _PUNCT_RE.sub(" ", text.lower()).split()


class BM25Index:
    """
    BM25Okapi index over chunk texts. Built from the exact same
    `documents` list (and in the exact same order) as the
    FAISSVectorStore, so result indices line up between the two.
    """

    def __init__(self):
        self.documents: List[Dict] = []
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_tokens: List[List[str]] = []

    def build(self, documents: List[Dict]):
        """
        documents: the same list of {"text":, "metadata":} dicts
        passed to FAISSVectorStore.add(). Order matters -- it's how
        BM25 and FAISS results get matched up later.
        """
        self.documents = documents
        self._corpus_tokens = [_tokenize(d["text"]) for d in documents]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Return up to top_k chunks ranked by BM25 score, in the same
        {"score":, "index":, "document":} shape FAISSVectorStore.search
        returns, so the two can be fused directly.
        """
        if self._bm25 is None or not self.documents:
            return []

        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append({
                "score": float(scores[idx]),
                "index": int(idx),
                "document": self.documents[idx],
            })

        return results