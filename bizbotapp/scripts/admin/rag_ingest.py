import os
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# === Path Setup ===

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DOCS_DIR = os.path.join(PROJECT_ROOT, "company_docs")
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "vector_store", "chroma_db")

# === Setup ChromaDB and Embedder ===

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
collection = client.get_or_create_collection(name="company_docs")

# === Ingestion Function ===

def ingest_documents():
    new_chunks_count = 0

    if not os.path.exists(DOCS_DIR):
        print(f"⚠️ Directory does not exist: {DOCS_DIR}")
        return 0

    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                embedding = embedder.encode(chunk).tolist()
                collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    ids=[f"{filename}-{i}"],
                    metadatas=[{"source": filename}]
                )
            new_chunks_count += len(chunks)

    print(f"✅ Ingested {new_chunks_count} new chunks.")
    return new_chunks_count
