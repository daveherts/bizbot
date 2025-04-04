import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# === Path configuration ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RAG_DB_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "vector_store", "chroma_db"))
BASE_MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "models", "base", "Llama-3.2-1B-Instruct"))
ADAPTER_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "models", "adapters", "llamaft", "checkpoint-20154"))

# === Load embedding model and Chroma vector store ===
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=RAG_DB_DIR, embedding_function=embed_model)

# === System prompt template for customer support tone ===
SYSTEM_PROMPT = (
    "You are a helpful customer support assistant. "
    "Use the following context to answer in less than 20 words:\n\n"
)

def retrieve_context(query: str, k: int = 3) -> str:
    """Retrieve top-k relevant context snippets from ChromaDB."""
    results = db.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

def generate_rag_prompt(query: str) -> str:
    """Assemble the full RAG prompt: system message + context + user query."""
    context = retrieve_context(query)
    return SYSTEM_PROMPT + context + f"\n\nUser: {query}\nAssistant:"

def load_rag_model():
    """Load the fine-tuned LLaMA 1B model with LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_DIR, device_map="auto", torch_dtype="auto")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, torch_dtype="auto")
    return tokenizer, model
