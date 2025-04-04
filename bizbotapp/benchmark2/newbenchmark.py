import json
import csv
from model_loader import load_model
from benchmetrics.latency import time_inference
from benchmetrics.tone_analysis import evaluate_tone
from utils import clean_response  # ✅ New import

# ✅ Models to test (base model selection phase)
MODELS_TO_TEST = [
    "llama-1b", "llama-3b", "phi-2", "phi-4-mini",
    "gemma-1b", "mistral-7b", "gemma-7b"
]

# ✅ File paths
QUESTIONS_PATH = "benchmark2/data/benchmark_questions.json"
TONE_SAMPLE_PATH = "benchmark2/data/bitext_sample.json"
RESULTS_CSV = "benchmark2/results/newbenchmark_latency.csv"
TONE_CSV = "benchmark2/results/newbenchmark_tone.csv"

# ✅ 20-word limit
MAX_TOKENS = 30

def run_latency_eval(model_key):
    tokenizer, model = load_model(model_key)
    print(f"⚙️ Running latency test for: {model_key}")

    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        for q in questions:
            prompt = q["question"]
            latency, _ = time_inference(model, tokenizer, prompt, max_tokens=MAX_TOKENS)
            writer.writerow([prompt, round(latency, 2), model_key])

def run_tone_eval(model_key):
    tokenizer, model = load_model(model_key)
    print(f"🎤 Evaluating tone for: {model_key}")

    with open(TONE_SAMPLE_PATH) as f:
        pairs = json.load(f)

    with open(TONE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        for item in pairs:
            prompt = item["instruction"]
            expected = item["response"]
            _, raw_output = time_inference(model, tokenizer, prompt, max_tokens=MAX_TOKENS)
            cleaned_output = clean_response(raw_output)  # ✅ Clean system response
            sim_score = evaluate_tone(cleaned_output, expected)
            writer.writerow([prompt, expected, cleaned_output, sim_score, model_key])

if __name__ == "__main__":
    with open(RESULTS_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["question", "latency", "model"])

    with open(TONE_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["input", "expected_response", "model_response", "similarity", "model"])

    for model_key in MODELS_TO_TEST:
        run_latency_eval(model_key)
        run_tone_eval(model_key)

    print("\n✅ Newbenchmark complete (latency + tone)")
