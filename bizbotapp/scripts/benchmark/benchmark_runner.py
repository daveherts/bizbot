import os
import time
import json
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import chromadb

# =================================
# Setup Paths
# =================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

BENCHMARK_PATH = os.path.join(SCRIPT_DIR, "benchmark_questions.json")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "benchmark_results.csv")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
BASE_MODEL_PATH = os.path.join(MODEL_DIR, "base", "Llama-3.2-1B-Instruct")
ADAPTER_PATH = os.path.join(MODEL_DIR, "adapters", "llamaft", "checkpoint-20154")
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "vector_store", "chroma_db")

# =================================
# Configuration Flags
# =================================
USE_FINE_TUNED = True
USE_RAG = True

# =================================
# Load Model
# =================================
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if USE_FINE_TUNED:
        model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_PATH,
            local_files_only=True,
            torch_dtype=torch.float16
        )
    else:
        model = base_model

    return tokenizer, model.to("cuda" if torch.cuda.is_available() else "cpu")

# =================================
# Setup RAG
# =================================
client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
collection = client.get_or_create_collection(name="company_docs")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve_context(query, top_k=3):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    chunks = results.get("documents", [[]])[0]
    return "\n".join(chunks) if chunks else ""

# =================================
# Format Prompt
# =================================
def format_prompt(user_input, rag_context=""):
    instructions = "You are a helpful customer service assistant for BrewBeans Co."
    return f"System: {instructions}\n\n{rag_context}\n\nUser: {user_input}\n\nAssistant:"

# =================================
# Benchmark Utilities
# =================================
def load_benchmark():
    with open(BENCHMARK_PATH, "r") as f:
        return json.load(f)

def evaluate_response(response, expected_keywords):
    matches = sum(kw.lower() in response.lower() for kw in expected_keywords)
    return matches / len(expected_keywords)

# =================================
# Main Benchmark Logic
# =================================
def benchmark():
    tokenizer, model = load_model()
    dataset = load_benchmark()

    with open(RESULTS_PATH, "w", newline="") as csvfile:
        fieldnames = ["question", "response", "latency", "accuracy", "pass", "model_type"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in dataset:
            question = item["question"]
            expected_keywords = item["expected_keywords"]

            start = time.time()
            context = retrieve_context(question) if USE_RAG else ""
            prompt = format_prompt(question, context)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.2
                )

            latency = time.time() - start
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            acc = evaluate_response(response, expected_keywords)

            # Thresholds
            pass_status = acc >= 0.8 and latency <= 3.0
            mark = "✅" if pass_status else "⚠️"
            model_type = (
                "fine-tuned-RAG" if USE_FINE_TUNED and USE_RAG else
                "fine-tuned" if USE_FINE_TUNED else
                "base"
            )

            print(f"Q: {question}\nA: {response}\nLatency: {latency:.2f}s | Accuracy: {acc:.2f} {mark}\n")

            writer.writerow({
                "question": question,
                "response": response,
                "latency": round(latency, 2),
                "accuracy": round(acc, 2),
                "pass": "yes" if pass_status else "no",
                "model_type": model_type
            })

# =================================
# Run Benchmark
# =================================
if __name__ == "__main__":
    benchmark()
