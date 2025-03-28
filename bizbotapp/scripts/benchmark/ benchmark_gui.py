import os
import time
import json
import csv
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import chromadb

# ========================
# Path Setup
# ========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

BENCHMARK_PATH = os.path.join(SCRIPT_DIR, "benchmark_questions.json")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "benchmark_results.csv")

BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "base", "Llama-3.2-1B-Instruct")
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models", "adapters", "llamaft", "checkpoint-20154")
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "vector_store", "chroma_db")

# ========================
# Setup: Vector Store & Embedder
# ========================
client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
collection = client.get_or_create_collection(name="company_docs")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ========================
# Load Model
# ========================
def load_model(use_fine_tuned):
    if use_fine_tuned:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_PATH,
            local_files_only=True,
            torch_dtype=torch.float16
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    return tokenizer, model

# ========================
# Utility Functions
# ========================
def retrieve_context(query):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    chunks = results.get("documents", [[]])[0]
    return "\n".join(chunks) if chunks else ""

def format_prompt(user_input, rag_context):
    system = "You are a helpful assistant for BrewBeans Co."
    return f"System: {system}\n\n{rag_context}\n\nUser: {user_input}\n\nAssistant:"

def evaluate_response(response, expected_keywords):
    matches = sum(kw.lower() in response.lower() for kw in expected_keywords)
    return matches / len(expected_keywords)

# ========================
# Benchmark Core Logic
# ========================
def run_benchmark(model_type):
    use_fine_tuned = model_type in ["Fine-Tuned", "Fine-Tuned + RAG"]
    use_rag = "RAG" in model_type

    tokenizer, model = load_model(use_fine_tuned)

    with open(BENCHMARK_PATH, "r") as f:
        dataset = json.load(f)

    results = []
    with open(RESULTS_PATH, "w", newline="") as csvfile:
        fieldnames = ["question", "response", "latency", "accuracy", "pass", "model_type"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in dataset:
            question = item["question"]
            expected_keywords = item["expected_keywords"]

            context = retrieve_context(question) if use_rag else ""
            prompt = format_prompt(question, context)

            start = time.time()
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.2)
            latency = time.time() - start

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            acc = evaluate_response(response, expected_keywords)
            passed = acc >= 0.8 and latency <= 3.0

            writer.writerow({
                "question": question,
                "response": response,
                "latency": round(latency, 2),
                "accuracy": round(acc, 2),
                "pass": "yes" if passed else "no",
                "model_type": model_type
            })

            results.append(f"✅" if passed else "⚠️" + f" Q: {question}\nA: {response}\nLatency: {latency:.2f}s | Accuracy: {acc:.2f}\n")

    return "\n---\n".join(results)

# ========================
# Gradio Interface
# ========================
with gr.Blocks() as demo:
    gr.Markdown("# 🧪 BizBot Benchmark Runner")

    model_choice = gr.Radio(
        choices=["Base", "Fine-Tuned", "Fine-Tuned + RAG"],
        label="Select Model Type",
        value="Fine-Tuned + RAG"
    )

    run_button = gr.Button("Run Benchmark")
    output_box = gr.Textbox(label="Benchmark Output", lines=20)

    run_button.click(fn=run_benchmark, inputs=model_choice, outputs=output_box)

demo.launch()
