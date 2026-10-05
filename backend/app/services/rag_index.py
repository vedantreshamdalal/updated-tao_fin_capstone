"""
Build the in-process RAG index + TAO pipeline from synced SEC filings.

Extracted from main.py so /api/analyze can rebuild the index after an
on-demand company sync, instead of requiring a manual sync + restart.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from app.db.database import SessionLocal
from app.db.models import Company, SECFiling
from app.services.bm25_index import BM25Index
from app.services.chunker import split_into_chunks
from app.services.embeddings import EmbeddingService
from app.services.hybrid_retriever import HybridRetriever
from app.services.ollama_client import OllamaClient
from app.services.rag_pipeline import RAGPipeline
from app.services.reranker import Reranker
from app.services.table_parser import extract_table_chunks
from app.services.tao_pipeline import TAOPipeline
from app.services.vector_store import FAISSVectorStore


def load_filing_records() -> List[Dict]:
    db = SessionLocal()
    try:
        filings = (
            db.query(SECFiling)
            .filter(SECFiling.local_text_path.isnot(None))
            .all()
        )
        return [
            {
                "text_path": Path(f.local_text_path),
                "html_path": Path(f.local_html_path) if f.local_html_path else None,
                "ticker": f.company.ticker,
                "form": f.form,
            }
            for f in filings
        ]
    finally:
        db.close()


def load_company_index() -> Dict[str, str]:
    db = SessionLocal()
    try:
        return {
            c.ticker: (c.name or c.ticker)
            for c in db.query(Company).all()
        }
    finally:
        db.close()


def build_tao_pipeline(
    filing_records: Optional[List[Dict]] = None,
) -> Tuple[Optional[TAOPipeline], Set[str]]:
    """
    Index every synced filing and return (pipeline, indexed_tickers).

    Returns (None, set()) when there is nothing to index. Does not
    raise merely because the corpus is empty.
    """

    records = filing_records if filing_records is not None else load_filing_records()

    if not records:
        return None, set()

    print(f"[index] Indexing {len(records)} filing(s) from the database...")

    embedding_service = EmbeddingService()
    all_chunks = []

    for record in records:
        if not record["text_path"].exists():
            print(f"[index] Skipping missing file: {record['text_path']}")
            continue

        text = record["text_path"].read_text(encoding="utf-8")
        all_chunks.extend(
            split_into_chunks(
                text=text,
                chunk_size=1200,
                overlap=200,
                metadata={
                    "source": "SEC EDGAR",
                    "company": record["ticker"],
                    "form": record["form"],
                    "filing": record["text_path"].name,
                    "type": "narrative",
                },
            )
        )

        if record["html_path"] and record["html_path"].exists():
            table_chunks = extract_table_chunks(
                str(record["html_path"]),
                metadata={
                    "source": "SEC EDGAR",
                    "company": record["ticker"],
                    "form": record["form"],
                    "filing": record["text_path"].name,
                },
            )
            all_chunks.extend(table_chunks)
            print(
                f"[index] Extracted {len(table_chunks)} table chunk(s) "
                f"from {record['html_path'].name}"
            )

    if not all_chunks:
        return None, set()

    embeddings = embedding_service.embed_documents(
        [c["text"] for c in all_chunks]
    )

    vector_store = FAISSVectorStore(dimension=embeddings.shape[1])
    vector_store.add(embeddings, all_chunks)

    bm25_index = BM25Index()
    bm25_index.build(all_chunks)

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedding_service=embedding_service,
        reranker=Reranker(),
    )

    rag_pipeline = RAGPipeline(
        retriever=retriever,
        ollama_client=OllamaClient(),
    )

    pipeline = TAOPipeline(
        rag_pipeline=rag_pipeline,
        company_index=load_company_index(),
    )

    indexed = {record["ticker"] for record in records}
    print(
        f"[index] Indexed {len(all_chunks)} chunks from {len(records)} filing(s) "
        f"(FAISS + BM25, cross-encoder rerank enabled)."
    )
    return pipeline, indexed
