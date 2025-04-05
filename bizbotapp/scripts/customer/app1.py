import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import chromadb

# --- Configurable Paths ---
BASE_MODEL_PATH = os.path.expanduser("~/bb/bizbotapp/scripts/models/base/Llama-3.2-1B-Instruct")
ADAPTER_PATH = os.path.expanduser("~/bb/bizbotapp/models/adapters/llamaft")
CHROMA_DB_PATH = os.path.expanduser("~/bb/bizbotapp/scripts/vector_store/chroma_db")

# --- Global vars ---
tokenizer = None
model = None
embedder = None
collection = None

# --- Load Embedder ---
def load_embedder():
    global embedder
    print("Loading Sentence Transformer model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedder loaded.")

# --- Load Vector DB ---
def init_vector_store():
    global collection
    print(f"Initializing ChromaDB client at: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    print("Getting or creating collection: company_docs")
    collection = client.get_or_create_collection(name="company_docs")

# --- Load LLM + Adapter ---
def load_bizbot():
    global tokenizer, model

    print("Loading tokenizer from:", BASE_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    print("Loading base model from:", BASE_MODEL_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

    print("Loading PEFT adapter from:", ADAPTER_PATH)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("Model with adapter loaded.")

# --- Answering logic ---
def answer_question(query):
    print("Embedding query for retrieval...")
    embedded = embedder.encode([query])
    print("Query embedded. Fetching relevant documents...")
    results = collection.query(query_texts=[query], n_results=3)
    print("ChromaDB results:", results)
    context = "\n".join(results['documents'][0]) if results['documents'] else ""

    prompt = f"Answer the following question based on the provided context:\nContext: {context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=256)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return answer.split("Answer:")[-1].strip()

# --- Launch Gradio UI ---
def launch_gradio():
    iface = gr.Interface(fn=answer_question, 
                         inputs=gr.Textbox(lines=2, label="Ask BizBot a question"),
                         outputs="text")
    iface.launch()

# --- Main logic ---
if __name__ == "__main__":
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    load_embedder()
    init_vector_store()
    load_bizbot()
    launch_gradio()