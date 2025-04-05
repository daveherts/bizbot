import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import gradio as gr
from rag.ingestion.ingest_from_files import ingest_documents
from chromadb import PersistentClient

VECTOR_STORE_PATH = os.path.expanduser("~/bb/bizbotapp/rag/vector_store/chroma_db")
DOCUMENTS_DIR = os.path.expanduser("~/bb/bizbotapp/company_docs")
client = PersistentClient(path=VECTOR_STORE_PATH)
collection = client.get_or_create_collection("company_docs")

def run_ingestion():
    try:
        ingest_documents()
        return "✅ Ingestion complete."
    except Exception as e:
        return f"❌ Ingestion failed: {str(e)}"

def list_docs():
    results = collection.get(include=["documents", "metadatas"])
    if not results["documents"]:
        return "⚠️ No documents found."

    lines = []
    for i, doc in enumerate(results["documents"]):
        meta = results["metadatas"][i]
        lines.append(f"[{i+1}] {meta.get('source', 'unknown')} (page {meta.get('page', '?')})")
    return "\n".join(lines)

def delete_docs():
    all_ids = collection.get()["ids"]
    if not all_ids:
        return "⚠️ No documents to delete."
    collection.delete(ids=all_ids)
    return "✅ All documents deleted."

with gr.Blocks(title="BizBot Admin Panel") as admin_ui:
    gr.Markdown("📁 **BizBot Admin Panel**")

    with gr.Row():
        file_upload = gr.File(label="Upload document", file_types=[".txt", ".pdf", ".md"])
        upload_status = gr.Textbox(label="Upload Status")

    with gr.Row():
        ingest_button = gr.Button("Ingest Documents")
        ingestion_status = gr.Textbox(label="Ingestion Status")

    with gr.Row():
        list_button = gr.Button("List Documents")
        list_output = gr.Textbox(label="Indexed Documents")

    with gr.Row():
        delete_button = gr.Button("Delete All Documents")
        delete_status = gr.Textbox(label="Deletion Status")

    def handle_upload(file):
        if file is None:
            return "❌ No file uploaded."
        filename = os.path.basename(file.name)
        dest = os.path.join(DOCUMENTS_DIR, filename)
        os.rename(file.name, dest)
        return f"✅ Uploaded: {filename}"

    file_upload.change(fn=handle_upload, inputs=file_upload, outputs=upload_status)
    ingest_button.click(fn=run_ingestion, outputs=ingestion_status)
    list_button.click(fn=list_docs, outputs=list_output)
    delete_button.click(fn=delete_docs, outputs=delete_status)

if __name__ == "__main__":
    admin_ui.launch(server_port=7861)
