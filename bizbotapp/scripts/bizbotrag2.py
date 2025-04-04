import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import chromadb
from sentence_transformers import SentenceTransformer
import gradio as gr

# Paths
MODEL_DIR = "/home/dave/project/bizbot/bizbot/models"
BASE_MODEL_PATH = f"{MODEL_DIR}/Llama-3.2-1B-Instruct"
ADAPTER_PATH = f"{MODEL_DIR}/llamaft/checkpoint-20154"
VECTOR_DB_DIR = "./vector_store/chroma_db"

device = "cuda" if torch.cuda.is_available() else "cpu"

current_model = None
current_tokenizer = None
conversation_history = []

# Embedded system instructions
INSTRUCTIONS = "You are a helpful customer service assistant for BrewBeans Co. Use the provided company information where possible to answer questions clearly and accurately."
MAX_TOKENS = 100
TEMPERATURE = 0.22

# Load vector DB and embedder
client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
collection = client.get_or_create_collection(name="company_docs")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def ensure_base_model():
    if not os.path.exists(BASE_MODEL_PATH):
        print("Base model not found. Downloading...")
        os.makedirs(BASE_MODEL_PATH, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.float16)
        tokenizer.save_pretrained(BASE_MODEL_PATH)
        model.save_pretrained(BASE_MODEL_PATH)
        print("Base model downloaded.")
    else:
        print("Base model found locally.")

def clear_model():
    global current_model, current_tokenizer
    if current_model:
        del current_model
        del current_tokenizer
        torch.cuda.empty_cache()
        current_model, current_tokenizer = None, None

def load_bizbot():
    global current_model, current_tokenizer
    clear_model()
    ensure_base_model()

    current_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    current_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        local_files_only=True,
        torch_dtype=torch.float16
    ).to(device)

    return "✅ BizBot fine-tuned model with RAG loaded."

def retrieve_context(query, top_k=3):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    chunks = results.get("documents", [[]])[0]
    if chunks:
        print("\n🔎 Retrieved RAG context:")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {chunk}\n")
    return "\n".join(chunks) if chunks else ""

def chat(user_input):
    global conversation_history

    if current_model is None or current_tokenizer is None:
        return "Model not loaded, please reload."

    rag_context = retrieve_context(user_input)

    conversation_history.append(f"User: {user_input}")
    formatted_input = (
        f"System: {INSTRUCTIONS}\n\n"
        f"Relevant company documentation to help answer this question:\n\n"
        f"{rag_context}\n\n"
        + "\n".join(conversation_history)
        + f"\n\nAssistant:"
    )

    inputs = current_tokenizer(formatted_input, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = current_model.generate(
            **inputs, max_new_tokens=MAX_TOKENS, temperature=TEMPERATURE
        )

    response = current_tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Assistant:")[-1].strip()

    conversation_history.append(f"Assistant: {response}")

    return response

# Auto-load on startup
load_bizbot()

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 BizBot RAG Chatbot")

    user_input = gr.Textbox(label="Ask BizBot a question")
    response_box = gr.Textbox(label="Response", interactive=False)

    submit_button = gr.Button("Submit")
    clear_chat_button = gr.Button("Clear Chat History")

    submit_button.click(chat, inputs=user_input, outputs=response_box)
    clear_chat_button.click(lambda: conversation_history.clear() or "", outputs=response_box)

demo.launch()
