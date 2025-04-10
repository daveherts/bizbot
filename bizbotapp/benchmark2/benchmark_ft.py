from model_loader_ft import load_model
from benchmetrics.accuracy import evaluate_keywords
from benchmetrics.latency import time_inference
from benchmetrics.tone_analysis import evaluate_tone
from utils import extract_assistant_reply  

MODELS_TO_TEST = ["fine-tuned-llama-1b"]

QUESTIONS_PATH = "benchmark2/data/benchmark_questions.json"
TONE_SAMPLE_PATH = "benchmark2/data/bitext_sample.json"
RESULTS_CSV = "benchmark2/results/full_results_ft.csv"
TONE_CSV = "benchmark2/results/tone_scores_ft.csv"

def run_single_eval(model_key):
    tokenizer, model = load_model(model_key)
    import json, csv
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    print(f"Running benchmark for: {model_key}")
    with open(RESULTS_CSV, "a", newline="") as file:
        writer = csv.writer(file)
        for q in questions:
            prompt = q["question"]
            expected = q["expected_keywords"]
            latency, raw_output = time_inference(model, tokenizer, prompt)
            cleaned_output = extract_assistant_reply(raw_output, prompt)
            accuracy = evaluate_keywords(cleaned_output, expected)
            writer.writerow([prompt, expected, cleaned_output, round(latency, 2), accuracy, model_key])

def run_tone_eval(model_key):
    tokenizer, model = load_model(model_key)
    import json, csv
    with open(TONE_SAMPLE_PATH) as f:
        pairs = json.load(f)

    print(f"Evaluating tone for: {model_key}")
    with open(TONE_CSV, "a", newline="") as file:
        writer = csv.writer(file)
        for item in pairs:
            prompt = item["instruction"]
            ref = item["response"]
            _, raw_output = time_inference(model, tokenizer, prompt)
            cleaned_output = extract_assistant_reply(raw_output, prompt)
            sim_score = evaluate_tone(cleaned_output, ref)
            writer.writerow([prompt, ref, cleaned_output, sim_score, model_key])

if __name__ == "__main__":
    import csv
    with open(RESULTS_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["question", "expected_keywords", "model_response", "latency", "accuracy", "model"])

    with open(TONE_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["input", "expected_response", "model_response", "similarity", "model"])

    for model_key in MODELS_TO_TEST:
        run_single_eval(model_key)
        run_tone_eval(model_key)
