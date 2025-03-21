# admin_gui.py
import gradio as gr
import shutil
import os
from scripts.rag_ingest import ingest_documents

UPLOAD_DIR = "./company_docs"

def handle_upload(file):
    # Move uploaded file into company_docs
    save_path = os.path.join(UPLOAD_DIR, file.name)
    shutil.move(file.name, save_path)
    return f"✅ Uploaded {file.name}"

def run_ingest():
    chunks_added = ingest_documents()
    return f"✅ Ingestion complete. {chunks_added} new chunks added to the vector DB."

with gr.Blocks() as admin_app:
    gr.Markdown("# 🛠️ BizBot Admin Panel")
    
    with gr.Row():
        upload = gr.File(label="Upload Company Document", file_types=[".txt"])
        upload_button = gr.Button("Upload Document")
    upload_status = gr.Textbox(label="Upload Status", interactive=False)

    ingest_button = gr.Button("Ingest All Documents into Vector Store")
    ingest_status = gr.Textbox(label="Ingestion Status", interactive=False)

    upload_button.click(handle_upload, inputs=[upload], outputs=upload_status)
    ingest_button.click(run_ingest, outputs=ingest_status)

admin_app.launch()
