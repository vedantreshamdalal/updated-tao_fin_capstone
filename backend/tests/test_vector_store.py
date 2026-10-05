
from app.services.embeddings import EmbeddingService
from app.services.vector_store import FAISSVectorStore


# --------------------------------------------------
# 1. Create embedding service
# --------------------------------------------------

embedding_service = EmbeddingService()


# --------------------------------------------------
# 2. Example financial documents
# --------------------------------------------------

documents = [
    {
        "text": (
            "Apple reported total net sales of "
            "$416,161 million in fiscal year 2025."
        ),
        "source": "Apple 2025 10-K",
        "section": "Statements of Operations"
    },
    {
        "text": (
            "Apple reported total net sales of "
            "$391,035 million in fiscal year 2024."
        ),
        "source": "Apple 2025 10-K",
        "section": "Statements of Operations"
    },
    {
        "text": (
            "Apple provides hardware products including "
            "iPhone, Mac and iPad."
        ),
        "source": "Apple 2025 10-K",
        "section": "Business"
    }
]


# --------------------------------------------------
# 3. Extract text
# --------------------------------------------------

texts = [
    document["text"]
    for document in documents
]


# --------------------------------------------------
# 4. Generate embeddings
# --------------------------------------------------

embeddings = (
    embedding_service.embed_documents(
        texts
    )
)


print()
print("Embedding shape:")
print(embeddings.shape)


# --------------------------------------------------
# 5. Create FAISS store
# --------------------------------------------------

vector_store = FAISSVectorStore(
    dimension=embeddings.shape[1]
)


# --------------------------------------------------
# 6. Add documents
# --------------------------------------------------

vector_store.add(
    embeddings,
    documents
)


print()
print("Documents in FAISS:")
print(vector_store.index.ntotal)


# --------------------------------------------------
# 7. Search
# --------------------------------------------------

query = (
    "What was Apple's total revenue in 2025?"
)

query_embedding = (
    embedding_service.embed_query(
        query
    )
)


results = vector_store.search(
    query_embedding,
    top_k=3
)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print()
print("=" * 80)
print("SEARCH RESULTS")
print("=" * 80)


for i, result in enumerate(
    results,
    start=1
):

    print()
    print(f"Result {i}")
    print("-" * 40)

    print(
        "Score:",
        result["score"]
    )

    print(
        "Source:",
        result["document"]["source"]
    )

    print(
        "Section:",
        result["document"]["section"]
    )

    print(
        "Text:",
        result["document"]["text"]
    )