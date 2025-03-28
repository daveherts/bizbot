import os
import time
import json
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === Flags ===
USE_FINE_TUNED = True
USE_RAG = True

# === File Paths ===
SCRIPT_DIR = os.path.dirname(__file__)
BENCHMARK_PATH = os.path.join(SCRIPT_DIR, "benchmark_questions.json")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "benchmark_results.csv")

# === Load Model ===
def load_model():
    if USE_FINE_TUNED:
        model_path = os.path.join(SCRIPT_DIR, "..", "models", "fine_tuned")
    else:
        model_path = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        device_map="auto"
    )
    return tokenizer, model

# === Dummy RAG Retriever ===
def retrieve_context(query):
    return "Relevant retrieved context for: " + query  # Replace with actual retriever

# === Prompt Formatting ===
def format_prompt(user_input, rag_context=""):
    instructions = "You are a helpful assistant for BrewBeans Co."
    return f"System: {instructions}\n\n{rag_context}\n\nUser: {user_input}\n\nAssistant:"

# === Load Benchmark Questions ===
def load_benchmark():
    with open(BENCHMARK_PATH, "r") as f:
        return json.load(f)

# === Keyword-Based Accuracy Scoring ===
def evaluate_response(response, expected_keywords):
    matches = sum(kw.lower() in response.lower() for kw in expected_keywords)
    return matches / len(expected_keywords)

# === Benchmark Runner ===
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
                    **inputs, max_new_tokens=100, temperature=0.2
                )
            latency = time.time() - start
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            acc = evaluate_response(response, expected_keywords)
            pass_status = acc >= 0.8 and latency <= 3.0
            mark = "✅" if pass_status else "⚠️"
            model_type = "fine-tuned-RAG" if USE_FINE_TUNED and USE_RAG else "base"

            print(f"\nQ: {question}\nA: {response}")
            print(f"Latency: {latency:.2f}s | Accuracy: {acc:.2f} {mark}")

            writer.writerow({
                "question": question,
                "response": response,
                "latency": round(latency, 2),
                "accuracy": round(acc, 2),
                "pass": "yes" if pass_status else "no",
                "model_type": model_type
            })

if __name__ == "__main__":
    benchmark()
