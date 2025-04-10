import os
from rag.ingestion.document_loader import load_document
from chromadb import PersistentClient

VECTOR_STORE_PATH = os.path.expanduser("~/bb/bizbotapp/rag/vector_store/chroma_db")
DOCUMENTS_DIR = os.path.expanduser("~/bb/bizbotapp/company_docs")

client = PersistentClient(path=VECTOR_STORE_PATH)
collection = client.get_or_create_collection("company_docs")

def ingest_documents():
    for filename in os.listdir(DOCUMENTS_DIR):
        path = os.path.join(DOCUMENTS_DIR, filename)
        if not os.path.isfile(path):
            continue

        print(f"Ingesting {filename}")
        chunks = load_document(path)

        collection.add(
            documents=[chunk.page_content for chunk in chunks],
            metadatas=[{"source": path, "page": i} for i, chunk in enumerate(chunks)],
            ids=[f"{filename}_{i}" for i in range(len(chunks))]
        )

    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_documents()
