import argparse
import csv
import json
import sys
import time
from pathlib import Path


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

BACKEND_DIR = (
    PROJECT_ROOT / "backend"
)

# PROJECT_ROOT makes "backend.*" / "evaluation.*" importable;
# BACKEND_DIR makes the "app.*" imports used inside the services
# resolve as well.
for _path in (
    PROJECT_ROOT,
    BACKEND_DIR,
):

    if str(_path) not in sys.path:

        sys.path.insert(
            0,
            str(_path)
        )


# =========================================================
# PROJECT IMPORTS
# =========================================================

from backend.app.services.financebench import (
    FinanceBench
)

from backend.app.services.embeddings import (
    EmbeddingService
)

from backend.app.services.ollama_client import (
    OllamaClient
)

from backend.app.services.rag_pipeline import (
    RAGPipeline
)

from backend.app.services.tao_pipeline import (
    TAOPipeline
)

from backend.app.services.vector_store import (
    FAISSVectorStore
)

from backend.app.services.bm25_index import (
    BM25Index
)

from backend.app.services.reranker import (
    Reranker
)

from backend.app.services.hybrid_retriever import (
    HybridRetriever
)

from backend.app.services.chunker import (
    split_into_chunks
)

from evaluation.financebench_evaluator import (
    evaluate_question,
    calculate_summary,
)


# =========================================================
# FINANCEBENCH PATH
# =========================================================

FINANCEBENCH_DIR = (
    PROJECT_ROOT
    / "Data"
    / "financebench"
)

PDF_DIR = (
    PROJECT_ROOT
    / "Data"
    / "pdfs"
)


# =========================================================
# SHARED MODELS
# =========================================================

# The embedding model, cross-encoder and Ollama client are stateless
# with respect to the document being indexed, so they are loaded once
# and reused across every per-document index.
_SHARED = {}


def _shared_embedding_service():

    if "embeddings" not in _SHARED:

        _SHARED["embeddings"] = (
            EmbeddingService()
        )

    return _SHARED["embeddings"]


def _shared_reranker():

    if "reranker" not in _SHARED:

        _SHARED["reranker"] = (
            Reranker()
        )

    return _SHARED["reranker"]


def _shared_ollama_client():

    if "ollama" not in _SHARED:

        _SHARED["ollama"] = (
            OllamaClient()
        )

    return _SHARED["ollama"]


# =========================================================
# BUILD DOCUMENT INDEX
# =========================================================

def build_financebench_index(
    doc_names=None
):
    """
    Build a retrieval index + TAO pipeline over the given FinanceBench
    documents.

    doc_names limits indexing to specific filings. Callers pass a
    single document: every FinanceBench question names the filing it
    must be answered from, so retrieval is scoped to that document
    rather than to a pooled corpus of all 84 filings. Pooling them
    would force the retriever to choose between near-identical cash
    flow statements from dozens of companies and years, which is a
    different and much harder task than the benchmark poses -- and
    indexing all 368 PDFs would take hours of CPU time per run.
    """

    print()
    print(
        "Building FinanceBench RAG index..."
    )

    if not PDF_DIR.exists():

        raise FileNotFoundError(
            f"""
FinanceBench PDF directory not found:

{PDF_DIR}

You currently have the two JSONL files, but
the financial-document PDFs are also required
for document-grounded RAG evaluation.
"""
        )

    if doc_names:

        pdf_files = []

        for doc_name in sorted(doc_names):

            path = (
                PDF_DIR
                / f"{doc_name}.pdf"
            )

            if not path.exists():

                raise FileNotFoundError(
                    f"No PDF named '{doc_name}.pdf' "
                    f"in {PDF_DIR}"
                )

            pdf_files.append(path)

    else:

        pdf_files = sorted(
            PDF_DIR.glob(
                "*.pdf"
            )
        )

    if not pdf_files:

        raise FileNotFoundError(
            f"""
No FinanceBench PDFs were found in:

{PDF_DIR}
"""
        )

    print(
        f"Found {len(pdf_files)} PDF files."
    )

    from pypdf import PdfReader

    all_chunks = []

    # -----------------------------------------------------
    # Extract PDFs page-by-page
    # -----------------------------------------------------

    for pdf_path in pdf_files:

        print(
            f"Reading: {pdf_path.name}"
        )

        try:

            reader = PdfReader(
                str(pdf_path)
            )

        except Exception as exc:

            print(
                f"Could not read "
                f"{pdf_path.name}: {exc}"
            )

            continue

        for page_number, page in enumerate(
            reader.pages
        ):

            try:

                text = (
                    page.extract_text()
                    or ""
                )

            except Exception:

                text = ""

            if not text.strip():

                continue

            chunks = split_into_chunks(
                text=text,
                chunk_size=1200,
                overlap=200,
                metadata={

                    "source":
                        "FinanceBench",

                    "filing":
                        pdf_path.name,

                    "document":
                        pdf_path.name,

                    "page":
                        page_number,

                    "page_number":
                        page_number,

                    "type":
                        "financebench",

                }
            )

            all_chunks.extend(
                chunks
            )

    if not all_chunks:

        raise RuntimeError(
            "No text chunks were extracted "
            "from FinanceBench PDFs."
        )

    print(
        f"Created {len(all_chunks)} chunks."
    )

    # -----------------------------------------------------
    # Embeddings
    # -----------------------------------------------------

    embedding_service = (
        _shared_embedding_service()
    )

    print(
        "Creating embeddings..."
    )

    embeddings = (
        embedding_service
        .embed_documents(
            [
                chunk["text"]
                for chunk in all_chunks
            ]
        )
    )

    # -----------------------------------------------------
    # FAISS
    # -----------------------------------------------------

    vector_store = (
        FAISSVectorStore(
            dimension=embeddings.shape[1]
        )
    )

    vector_store.add(
        embeddings,
        all_chunks
    )

    # -----------------------------------------------------
    # BM25
    # -----------------------------------------------------

    bm25_index = BM25Index()

    bm25_index.build(
        all_chunks
    )

    # -----------------------------------------------------
    # Reranker
    # -----------------------------------------------------

    reranker = _shared_reranker()

    # -----------------------------------------------------
    # Hybrid retriever
    # -----------------------------------------------------

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedding_service=embedding_service,
        reranker=reranker,
    )

    # -----------------------------------------------------
    # Ollama
    # -----------------------------------------------------

    ollama_client = (
        _shared_ollama_client()
    )

    # -----------------------------------------------------
    # RAG
    # -----------------------------------------------------

    rag_pipeline = RAGPipeline(
        retriever=retriever,
        ollama_client=ollama_client,
    )

    # -----------------------------------------------------
    # TAO
    #
    # IMPORTANT:
    # No company index and no external fallback.
    # -----------------------------------------------------

    tao_pipeline = TAOPipeline(
        rag_pipeline=rag_pipeline,
        company_index={},
        allow_external_fallback=False,
    )

    print(
        "FinanceBench RAG index ready."
    )

    return tao_pipeline


# =========================================================
# RUN ONE QUESTION
# =========================================================

def run_question(
    pipeline,
    item
):

    question = (
        item.get(
            "question",
            ""
        )
    )

    start = time.perf_counter()

    try:

        result = pipeline.run(
            question
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        if not isinstance(
            result,
            dict
        ):

            result = {
                "answer":
                    str(result)
            }

        result[
            "latency_seconds"
        ] = round(
            elapsed,
            4
        )

        return result

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - start
        )

        return {

            "answer":
                "",

            "evidence":
                [],

            "verification":
                None,

            "latency_seconds":
                round(
                    elapsed,
                    4
                ),

            "error":
                str(exc)
        }


# =========================================================
# SAVE JSON
# =========================================================

def save_json(
    path,
    data
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# =========================================================
# SAVE CSV
# =========================================================

def save_csv(
    path,
    results
):

    if not results:
        return

    fields = [

        "financebench_id",

        "question",

        "company",

        "document",

        "question_type",

        "question_reasoning",

        "gold_answer",

        "predicted_answer",

        "answer_correct",

        "hit_at_1",

        "hit_at_3",

        "hit_at_5",

        "latency_seconds",

        "error",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for result in results:

            writer.writerow({
                field:
                    result.get(
                        field
                    )
                for field in fields
            })


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Load FinanceBench
    # -----------------------------------------------------

    print(
        "Loading FinanceBench..."
    )

    benchmark = FinanceBench(
        data_dir=str(
            FINANCEBENCH_DIR
        )
    )

    questions = (
        benchmark.get_questions()
    )

    print(
        f"Loaded {len(questions)} "
        f"FinanceBench questions."
    )

    # -----------------------------------------------------
    # Select questions
    # -----------------------------------------------------

    start = args.start

    if args.limit is None:

        end = len(
            questions
        )

    else:

        end = min(
            start + args.limit,
            len(questions)
        )

    questions = questions[
        start:end
    ]

    print(
        f"Running questions "
        f"{start + 1} to {end}."
    )

    # -----------------------------------------------------
    # Build TAO
    # -----------------------------------------------------

    # Indexes are built lazily and cached per document, so a run only
    # ever embeds the filings its questions actually reference.
    pipelines = {}

    def get_pipeline(
        doc_name
    ):

        if doc_name not in pipelines:

            pipelines[doc_name] = (
                build_financebench_index(
                    doc_names={doc_name}
                )
            )

        return pipelines[doc_name]

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    results = []

    for index, item in enumerate(
        questions,
        start=start + 1
    ):

        fb_id = item.get(
            "financebench_id"
        )

        question = item.get(
            "question"
        )

        print()
        print(
            "=" * 70
        )

        print(
            f"[{index}/{len(benchmark.questions)}]"
        )

        print(
            f"ID: {fb_id}"
        )

        print(
            f"Question: {question}"
        )

        try:

            pipeline = get_pipeline(
                item.get("doc_name")
            )

        except Exception as exc:

            print(
                f"Could not index "
                f"{item.get('doc_name')}: {exc}"
            )

            model_result = {
                "answer": "",
                "evidence": [],
                "verification": None,
                "latency_seconds": None,
                "error": str(exc),
            }

        else:

            model_result = run_question(
                pipeline,
                item
            )

        evaluated = (
            evaluate_question(
                item,
                model_result
            )
        )

        results.append(
            evaluated
        )

        print(
            f"TAO Answer: "
            f"{evaluated['predicted_answer']}"
        )

        print(
            f"Correct: "
            f"{evaluated['answer_correct']}"
        )

        print(
            f"Hit@5: "
            f"{evaluated['hit_at_5']}"
        )

        print(
            f"Latency: "
            f"{evaluated['latency_seconds']} sec"
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary = (
        calculate_summary(
            results
        )
    )

    output_dir = (
        PROJECT_ROOT
        / "evaluation"
        / "results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    save_json(
        output_dir
        / "financebench_results.json",
        results
    )

    save_json(
        output_dir
        / "financebench_summary.json",
        summary
    )

    save_csv(
        output_dir
        / "financebench_results.csv",
        results
    )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print()
    print(
        "#" * 70
    )

    print(
        "FINANCEBENCH SUMMARY"
    )

    print(
        "#" * 70
    )

    print(
        f"Questions: "
        f"{summary['num_questions']}"
    )

    print(
        f"Answer Accuracy: "
        f"{summary['answer_accuracy']:.2%}"
    )

    print(
        f"Evidence Hit@1: "
        f"{summary['evidence_hit_at_1']:.2%}"
    )

    print(
        f"Evidence Hit@3: "
        f"{summary['evidence_hit_at_3']:.2%}"
    )

    print(
        f"Evidence Hit@5: "
        f"{summary['evidence_hit_at_5']:.2%}"
    )

    if (
        "average_latency_seconds"
        in summary
    ):

        print(
            f"Average latency: "
            f"{summary['average_latency_seconds']:.2f}s"
        )

    print(
        "#" * 70
    )


if __name__ == "__main__":

    main()