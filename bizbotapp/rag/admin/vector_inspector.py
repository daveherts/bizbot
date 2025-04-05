# rag/admin/vector_inspector.py

import os
from chromadb import PersistentClient

VECTOR_STORE_PATH = os.path.expanduser("~/bb/bizbotapp/scripts/vector_store/chroma_db")
client = PersistentClient(path=VECTOR_STORE_PATH)
collection = client.get_or_create_collection("company_docs")

def list_docs():
    print("📄 Listing all documents in 'company_docs' collection:")
    results = collection.get(include=["documents", "metadatas"])
    if not results["documents"]:
        print("⚠️ No documents found.")
        return

    for i, doc in enumerate(results["documents"]):
        meta = results["metadatas"][i]
        print(f"[{i+1}] {meta.get('source', 'unknown')} (page {meta.get('page', '?')})")

def clear_docs():
    all_ids = collection.get()["ids"]
    if not all_ids:
        print("⚠️ No documents to delete.")
        return

    confirm = input("⚠️ Are you sure you want to delete all documents? (yes/no): ")
    if confirm.lower() == "yes":
        collection.delete(ids=all_ids)
        print("✅ All documents deleted.")
    else:
        print("❌ Deletion cancelled.")

if __name__ == "__main__":
    print("1. List documents")
    print("2. Clear all\n")
    choice = input("Choose an option: ")

    if choice == "1":
        list_docs()
    elif choice == "2":
        clear_docs()
    else:
        print("Invalid option.")
