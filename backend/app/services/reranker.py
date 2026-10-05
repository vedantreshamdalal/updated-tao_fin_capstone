"""
Cross-encoder reranker for retrieved financial document chunks.

After hybrid retrieval (FAISS + BM25, fused via RRF) produces a
shortlist of candidates, the reranker rescores each (query, chunk)
pair with a cross-encoder that sees both texts jointly -- much more
accurate than the bi-encoder cosine similarity FAISS uses, at the
cost of being O(n) per query. That's fine because it only ever runs
on a small shortlist (~15-20 candidates), never the full corpus.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good
quality, ~80MB). Override via the RERANKER_MODEL env var.
"""

import os
from typing import Dict, List, Optional

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Cross-encoder reranker that rescores (query, chunk) pairs."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("RERANKER_MODEL", _DEFAULT_MODEL)
        self._model = None

    @property
    def model(self):
        # Lazy-loaded: the model only downloads/loads on first actual
        # use, not at import time or app startup.
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        candidates: list of {"score":, "index":, "document": {"text":, "metadata":}}
        (the shape produced by HybridRetriever's RRF fusion step).

        Returns the top_k candidates re-sorted by cross-encoder score
        (descending), with "score" replaced by that cross-encoder score.
        """
        if not candidates:
            return []

        pairs = [(query, c["document"]["text"]) for c in candidates]
        ce_scores = self.model.predict(pairs)

        rescored = []
        for cand, score in zip(candidates, ce_scores):
            rescored.append({
                "score": float(score),
                "index": cand["index"],
                "document": cand["document"],
            })

        rescored.sort(key=lambda c: c["score"], reverse=True)

        return rescored[:top_k]