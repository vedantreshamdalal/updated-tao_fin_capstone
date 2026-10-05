
from pathlib import Path

from app.services.chunker import split_into_chunks
from app.services.embeddings import EmbeddingService
from app.services.vector_store import FAISSVectorStore


# --------------------------------------------------
# 1. Locate SEC text
# --------------------------------------------------

sec_folder = Path("data/sec")

text_files = list(
    sec_folder.glob("*.txt")
)

if not text_files:
    raise FileNotFoundError(
        "No SEC text file found in data/sec/"
    )

text_file = text_files[0]

print("Loading:")
print(text_file)


# --------------------------------------------------
# 2. Read filing
# --------------------------------------------------

text = text_file.read_text(
    encoding="utf-8"
)

print()
print("Characters:", len(text))


# --------------------------------------------------
# 3. Create chunks
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

print()
print("Total chunks:", len(chunks))


# --------------------------------------------------
# 4. Create embeddings
# --------------------------------------------------

embedding_service = EmbeddingService()

chunk_texts = [
    chunk["text"]
    for chunk in chunks
]

print()
print("Creating embeddings...")

embeddings = (
    embedding_service.embed_documents(
        chunk_texts
    )
)

print()
print("Embedding shape:")
print(embeddings.shape)


# --------------------------------------------------
# 5. Create FAISS
# --------------------------------------------------

vector_store = FAISSVectorStore(
    dimension=embeddings.shape[1]
)


# --------------------------------------------------
# 6. Add chunks
# --------------------------------------------------

vector_store.add(
    embeddings,
    chunks
)

print()
print(
    "Vectors stored:",
    vector_store.index.ntotal
)


# --------------------------------------------------
# 7. Ask a financial question
# --------------------------------------------------

query = (
    "What was Apple's total net sales "
    "in fiscal year 2025?"
)

print()
print("Question:")
print(query)


query_embedding = (
    embedding_service.embed_query(
        query
    )
)


# --------------------------------------------------
# 8. Retrieve evidence
# --------------------------------------------------

results = vector_store.search(
    query_embedding,
    top_k=5
)


# --------------------------------------------------
# 9. Display evidence
# --------------------------------------------------

print()
print("=" * 80)
print("RETRIEVED EVIDENCE")
print("=" * 80)


for i, result in enumerate(
    results,
    start=1
):

    document = result["document"]

    print()
    print(f"RESULT {i}")
    print("-" * 80)

    print(
        "Similarity:",
        result["score"]
    )

    print(
        "Metadata:",
        document["metadata"]
    )

    print()
    print(document["text"])
