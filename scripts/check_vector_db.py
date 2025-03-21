import chromadb

client = chromadb.PersistentClient(path="./vector_store/chroma_db")
collection = client.get_or_create_collection(name="company_docs")

all_items = collection.get(include=["documents", "metadatas", "ids"])

print(f"Total records: {len(all_items['ids'])}")
for i in range(len(all_items['ids'])):
    print(f"ID: {all_items['ids'][i]}")
    print(f"Source file: {all_items['metadatas'][i].get('source')}")
    print(f"Chunk: {all_items['documents'][i][:80]}...")  # Print first 80 chars
