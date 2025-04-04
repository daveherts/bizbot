import json
import csv
from model_loader_ft import load_model
from benchmetrics.latency import time_inference
from benchmetrics.tone_analysis import evaluate_tone
from utils import clean_response

MODELS_TO_TEST = ["fine-tuned-llama-1b"]

QUESTIONS_PATH = "benchmark2/data/benchmark_questions.json"
TONE_SAMPLE_PATH = "benchmark2/data/bitext_sample.json"
LATENCY_CSV = "benchmark2/results/newbenchmark_ft_latency.csv"
TONE_CSV = "benchmark2/results/newbenchmark_ft_tone.csv"
MAX_TOKENS = 30

def run_latency_eval(model_key):
    tokenizer, model = load_model(model_key)
    print(f"Running latency test for: {model_key}")

    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    with open(LATENCY_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        for q in questions:
            prompt = q["question"]
            latency, _ = time_inference(model, tokenizer, prompt, max_tokens=MAX_TOKENS)
            writer.writerow([prompt, round(latency, 2), model_key])

def run_tone_eval(model_key):
    tokenizer, model = load_model(model_key)
    print(f"Evaluating tone for: {model_key}")

    with open(TONE_SAMPLE_PATH) as f:
        samples = json.load(f)

    with open(TONE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        for item in samples:
            prompt = item["instruction"]
            expected = item["response"]
            _, raw_output = time_inference(model, tokenizer, prompt, max_tokens=MAX_TOKENS)
            output = clean_response(raw_output, prompt)
            similarity = evaluate_tone(output, expected)
            writer.writerow([prompt, expected, output, similarity, model_key])

if __name__ == "__main__":
    with open(LATENCY_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["question", "latency", "model"])

    with open(TONE_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["input", "expected_response", "model_response", "similarity", "model"])

    for model_key in MODELS_TO_TEST:
        run_latency_eval(model_key)
        run_tone_eval(model_key)

    print("newbenchmark_ft complete (latency + tone only)")
