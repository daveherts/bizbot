from chromadb import PersistentClient

# Path to your ChromaDB
CHROMA_DB_PATH = "/home/dave/bb/bizbotapp/scripts/vector_store/chroma_db"
client = PersistentClient(path=CHROMA_DB_PATH)

# Create or get the collection
collection = client.get_or_create_collection("company_docs")

# Optional: Clear everything using a dummy filter
try:
    collection.delete(where={"doc_type": "support"})
except Exception as e:
    print("⚠️ Couldn't delete existing docs:", e)

# Add support-related content with metadata
docs = [
    "Our customer support team is available from 9am to 6pm, Monday through Friday.",
    "You can contact us at support@example.com or by calling 0800 123 456.",
    "Visit our help center at https://support.example.com for FAQs and live chat support."
]

metadatas = [{"doc_type": "support"} for _ in docs]

collection.add(
    documents=docs,
    metadatas=metadatas,
    ids=[f"doc-{i}" for i in range(len(docs))]
)

print("📥 Ingestion complete! Added", len(docs), "documents.")
