import json
import csv
from model_loader_ft import load_model
from benchmetrics.latency import time_inference
from benchmetrics.accuracy import evaluate_keywords
from benchmetrics.tone_analysis import evaluate_tone
from utils import clean_response  # ✅ ensure this is available

# ✅ One model key for fine-tuned model
MODELS_TO_TEST = ["fine-tuned-llama-1b"]

# ✅ Benchmark files
QUESTIONS_PATH = "benchmark2/data/benchmark_questions.json"
TONE_SAMPLE_PATH = "benchmark2/data/bitext_sample.json"
RESULTS_CSV = "benchmark2/results/newbenchmark_ft_latency_accuracy.csv"
TONE_CSV = "benchmark2/results/newbenchmark_ft_tone.csv"
MAX_TOKENS = 30

def run_accuracy_eval(model_key):
    tokenizer, model = load_model(model_key)
    print(f"⚙️ Running latency + accuracy for: {model_key}")

    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        for q in questions:
            prompt = q["question"]
            expected = q["expected_keywords"]
            latency, raw_output = time_inference(model, tokenizer, prompt, max_tokens=MAX_TOKENS)
            cleaned_output = clean_response(raw_output)
            accuracy = evaluate_keywords(cleaned_output, expected)
            writer.writerow([prompt, cleaned_output, round(latency, 2), accuracy, model_key])

def run_tone_eval(model_key):
    tokenizer, model = load_model(model_key)
    print(f"🎤 Evaluating tone for: {model_key}")

    with open(TONE_SAMPLE_PATH) as f:
        samples = json.load(f)

    with open(TONE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        for item in samples:
            prompt = item["instruction"]
            expected = item["response"]
            _, raw_output = time_inference(model, tokenizer, prompt, max_tokens=MAX_TOKENS)
            cleaned_output = clean_response(raw_output)
            similarity = evaluate_tone(cleaned_output, expected)
            writer.writerow([prompt, expected, cleaned_output, similarity, model_key])

if __name__ == "__main__":
    with open(RESULTS_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["question", "model_response", "latency", "accuracy", "model"])

    with open(TONE_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["input", "expected_response", "model_response", "similarity", "model"])

    for model_key in MODELS_TO_TEST:
        run_accuracy_eval(model_key)
        run_tone_eval(model_key)

    print("\n✅ newbenchmark_ft complete (latency + accuracy + tone)")
