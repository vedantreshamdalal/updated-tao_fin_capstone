
from app.services.embeddings import EmbeddingService


embedding_service = EmbeddingService()


documents = [
    "Apple reported total net sales of $416,161 million in 2025.",
    "Apple's services revenue increased during fiscal year 2025.",
    "The company operates in the technology industry."
]


print()
print("Creating document embeddings...")

document_embeddings = (
    embedding_service.embed_documents(
        documents
    )
)

print()
print("Embedding shape:")
print(document_embeddings.shape)


query = (
    "What was Apple's total revenue in 2025?"
)

print()
print("Creating query embedding...")

query_embedding = (
    embedding_service.embed_query(
        query
    )
)

print()
print("Query embedding shape:")
print(query_embedding.shape)