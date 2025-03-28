import os
import time
import json
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Flags
USE_FINE_TUNED = True
USE_RAG = True

# Paths
BENCHMARK_PATH = "benchmark_set.json"
RESULTS_PATH = "benchmark_results.csv"

# Load model/tokenizer (toggle between base or fine-tuned model)
def load_model():
    if USE_FINE_TUNED:
        # Load your fine-tuned BizBot model here
        # Replace with actual loading code (LoRA, PEFT, etc.)
        model_path = "../models/fine_tuned"
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

# Dummy RAG retriever function
def retrieve_context(query):
    return "Relevant retrieved context for: " + query  # Replace with actual RAG call

def format_prompt(user_input, rag_context=""):
    instructions = "You are a helpful assistant for BrewBeans Co."
    return f"System: {instructions}\n\n{rag_context}\n\nUser: {user_input}\n\nAssistant:"

def load_benchmark():
    with open(BENCHMARK_PATH, "r") as f:
        return json.load(f)

def evaluate_response(response, expected_keywords):
    matches = sum(kw.lower() in response.lower() for kw in expected_keywords)
    return matches / len(expected_keywords)

def benchmark():
    tokenizer, model = load_model()
    dataset = load_benchmark()

    with open(RESULTS_PATH, "w", newline="") as csvfile:
        fieldnames = ["question", "response", "latency", "accuracy", "pass"]
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

            # Thresholds
            pass_status = acc >= 0.8 and latency <= 3.0
            mark = "✅" if pass_status else "⚠️"

            print(f"Q: {question}\nA: {response}\nLatency: {latency:.2f}s | Accuracy: {acc:.2f} {mark}\n")

            writer.writerow({
                "question": question,
                "response": response,
                "latency": round(latency, 2),
                "accuracy": round(acc, 2),
                "pass": "yes" if pass_status else "no"
            })

if __name__ == "__main__":
    benchmark()
