import gradio as gr
import shutil
import os
import chromadb
import sys

# Add project root to sys.path to allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rag_ingest import ingest_documents


# Set paths relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "company_docs")
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "vector_store", "chroma_db")

# Set up ChromaDB client and collection
client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
collection = client.get_or_create_collection(name="company_docs")

def handle_upload(file):
    if not file:
        return "⚠️ No file uploaded."

    original_filename = file.name.split("/")[-1]
    save_path = os.path.join(UPLOAD_DIR, original_filename)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    shutil.move(file.name, save_path)

    return f"✅ Uploaded and saved {original_filename} to company_docs/"

def run_ingest():
    chunks_added = ingest_documents()
    return f"✅ Ingestion complete. {chunks_added} new chunks added to the vector DB."

def show_vector_db_contents():
    all_items = collection.get(include=["documents", "metadatas"])
    count = len(all_items["ids"])
    if count == 0:
        return "⚠️ The vector database is currently empty."
    else:
        summary = f"✅ Current vector DB has {count} chunks:\n\n"
        for i in range(min(count, 5)):
            source = all_items['metadatas'][i].get('source')
            preview = all_items['documents'][i][:80]
            summary += f"- From **{source}**: \"{preview}...\"\n"
        if count > 5:
            summary += f"\n... and {count - 5} more chunks."
        return summary

def remove_from_vector_db(filename):
    all_items = collection.get(include=["metadatas"])
    ids_to_delete = [
        item_id for item_id, meta in zip(all_items["ids"], all_items["metadatas"])
        if meta.get("source") == filename
    ]

    if not ids_to_delete:
        return f"⚠️ No chunks found in vector DB from file: {filename}"

    collection.delete(ids=ids_to_delete)
    return f"🗑️ Removed {len(ids_to_delete)} chunks from vector DB for file: {filename}"

# Gradio UI
with gr.Blocks() as admin_app:
    gr.Markdown("# 🛠️ BizBot Admin Panel")

    with gr.Row():
        upload = gr.File(label="Upload Company Document", file_types=[".txt"], interactive=True)
        upload_button = gr.Button("Upload Document")
    upload_status = gr.Textbox(label="Upload Status", interactive=False)

    ingest_button = gr.Button("Ingest All Documents into Vector Store")
    ingest_status = gr.Textbox(label="Ingestion Status", interactive=False)

    check_db_button = gr.Button("Show Current Vector DB Contents")
    db_contents_box = gr.Markdown()

    with gr.Row():
        delete_filename = gr.Textbox(label="Filename to Remove from Vector DB (e.g., policy.txt)")
        delete_button = gr.Button("Remove File Chunks from Vector DB")
    delete_status = gr.Textbox(label="Deletion Result", interactive=False)

    upload_button.click(handle_upload, inputs=[upload], outputs=upload_status)
    ingest_button.click(run_ingest, outputs=ingest_status)
    check_db_button.click(show_vector_db_contents, outputs=db_contents_box)
    delete_button.click(remove_from_vector_db, inputs=delete_filename, outputs=delete_status)

admin_app.launch()
