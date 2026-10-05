
from pathlib import Path

from app.services.chunker import split_into_chunks
from app.services.embeddings import EmbeddingService
from app.services.vector_store import FAISSVectorStore
from app.services.ollama_client import OllamaClient
from app.services.rag_pipeline import RAGPipeline


# --------------------------------------------------
# 1. Load SEC filing
# --------------------------------------------------

sec_folder = Path("data/sec")

text_files = list(
    sec_folder.glob("*.txt")
)

if not text_files:
    raise FileNotFoundError(
        "No SEC text file found."
    )

text_file = text_files[0]

text = text_file.read_text(
    encoding="utf-8"
)


# --------------------------------------------------
# 2. Chunk document
# --------------------------------------------------

chunks = split_into_chunks(
    text=text,
    chunk_size=1200,
    overlap=200,
    metadata={
        "company": "Apple Inc.",
        "cik": "0000320193",
        "form": "10-K",
        "source": "SEC EDGAR",
        "filing": text_file.name
    }
)

print("Chunks:", len(chunks))


# --------------------------------------------------
# 3. Embeddings
# --------------------------------------------------

embedding_service = EmbeddingService()

chunk_texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = (
    embedding_service.embed_documents(
        chunk_texts
    )
)


# --------------------------------------------------
# 4. FAISS
# --------------------------------------------------

vector_store = FAISSVectorStore(
    dimension=embeddings.shape[1]
)

vector_store.add(
    embeddings,
    chunks
)


# --------------------------------------------------
# 5. Ollama
# --------------------------------------------------

ollama_client = OllamaClient(
    model="phi3:latest"
)


# --------------------------------------------------
# 6. RAG Pipeline
# --------------------------------------------------

rag = RAGPipeline(
    vector_store=vector_store,
    embedding_service=embedding_service,
    ollama_client=ollama_client
)


# --------------------------------------------------
# 7. Ask question
# --------------------------------------------------

question = (
    "What was Apple's total net sales "
    "in fiscal year 2025?"
)

print()
print("=" * 80)
print("QUESTION")
print("=" * 80)

print(question)


# --------------------------------------------------
# 8. Generate answer
# --------------------------------------------------

result = rag.answer(
    question,
    top_k=2
)


# --------------------------------------------------
# 9. Display answer
# --------------------------------------------------

print()
print("=" * 80)
print("ANSWER")
print("=" * 80)

print(result["answer"])


# --------------------------------------------------
# 10. Display evidence
# --------------------------------------------------

print()
print("=" * 80)
print("EVIDENCE USED")
print("=" * 80)

for i, evidence in enumerate(
    result["evidence"],
    start=1
):

    print()
    print(f"EVIDENCE {i}")
    print("-" * 80)

    print(
        "Similarity:",
        evidence["score"]
    )

    print(
        evidence["document"]["text"][:1000]
    )
