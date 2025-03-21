import gradio as gr
import shutil
import os
import chromadb
from scripts.rag_ingest import ingest_documents

UPLOAD_DIR = "./company_docs"
VECTOR_DB_DIR = "./vector_store/chroma_db"
client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
collection = client.get_or_create_collection(name="company_docs")


def handle_upload(file):
    if not file:
        return "⚠️ No file uploaded."

    # Extract the original filename
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
        for i in range(min(count, 5)):  # Show first 5 records only
            source = all_items['metadatas'][i].get('source')
            preview = all_items['documents'][i][:80]
            summary += f"- From **{source}**: \"{preview}...\"\n"
        if count > 5:
            summary += f"\n... and {count - 5} more chunks."
        return summary


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

    upload_button.click(handle_upload, inputs=[upload], outputs=upload_status)
    ingest_button.click(run_ingest, outputs=ingest_status)
    check_db_button.click(show_vector_db_contents, outputs=db_contents_box)

admin_app.launch()
