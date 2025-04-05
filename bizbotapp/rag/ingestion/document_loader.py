# ingestion/document_loader.py

import os
import fitz  # PyMuPDF

def load_document(path: str) -> str:
    ext = os.path.splitext(path)[-1].lower()

    try:
        if ext in [".txt", ".md"]:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

        elif ext == ".pdf":
            return extract_text_from_pdf(path)

    except Exception as e:
        print(f"⚠️ Failed to load {path}: {e}")
        return ""

    return ""

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text
