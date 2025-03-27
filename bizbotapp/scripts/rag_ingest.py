# scripts/rag_ingest.py
import os
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

VECTOR_DB_DIR = "./vector_store/chroma_db"
DOCS_DIR = "./company_docs"

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
collection = client.get_or_create_collection(name="company_docs")

def ingest_documents():
    new_chunks_count = 0
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_DIR, filename)
            with open(filepath, "r") as f:
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
    return new_chunks_count
