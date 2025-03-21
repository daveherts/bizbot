import chromadb

client = chromadb.PersistentClient(path="./vector_store/chroma_db")
collection = client.get_or_create_collection(name="company_docs")

# No need to request ids; they are always returned
all_items = collection.get(include=["documents", "metadatas"])

print(f"✅ Total records: {len(all_items['ids'])}")
for i in range(len(all_items['ids'])):
    print(f"\n➡️  ID: {all_items['ids'][i]}")
    print(f"   Source: {all_items['metadatas'][i].get('source')}")
    print(f"   Chunk preview: {all_items['documents'][i][:100]}...")
