# admin/vector_inspector.py

import os
from chromadb import PersistentClient

CHROMA_DB_PATH = os.path.expanduser("~/bb/bizbotapp/scripts/vector_store/chroma_db")

client = PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection("company_docs")

def list_docs():
    print("📄 Listing all documents in 'company_docs' collection:")
    results = collection.peek()  # Show a preview of items (default: 10)
    for i, doc in enumerate(results['documents']):
        meta = results['metadatas'][i]
        print(f"[{i+1}] {meta.get('filename', 'unknown')}\n    -> {doc[:150].strip()}...")

def clear_docs():
    confirm = input("⚠️ Are you sure you want to delete all documents? (yes/no): ")
    if confirm.lower() == "yes":
        all_ids = [doc_id for doc_id in collection.get()['ids']]
        collection.delete(ids=all_ids)
        print("✅ All documents deleted.")
    else:
        print("❌ Deletion cancelled.")

if __name__ == "__main__":
    print("1. List documents\n2. Clear all\n")
    choice = input("Choose an option: ")
    if choice == "1":
        list_docs()
    elif choice == "2":
        clear_docs()
    else:
        print("Invalid choice.")
