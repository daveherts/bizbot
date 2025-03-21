import chromadb

client = chromadb.PersistentClient(path="./vector_store/chroma_db")
collection = client.get_or_create_collection(name="company_docs")

all_items = collection.get(include=["documents", "metadatas"])
print(f"Total chunks in DB: {len(all_items['ids'])}")

for i, doc in enumerate(all_items['documents']):
    print(f"\nChunk {i}: {doc[:200]}")  # print first 200 chars
