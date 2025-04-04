import os
import json
import csv
from .ragmodel import load_rag_model, generate_rag_prompt
from .benchmetrics.accuracy import evaluate_keywords
from .benchmetrics.tone_analysis import evaluate_tone
from .benchmetrics.latency import time_inference
from .utils import extract_assistant_reply  # ✅ Clean model responses

# File paths
QUESTIONS_PATH = "benchmark2/data/benchmark_questions.json"
TONE_SAMPLE_PATH = "benchmark2/data/bitext_sample.json"
RESULTS_CSV = "benchmark2/results/rag_full_results.csv"
TONE_CSV = "benchmark2/results/rag_tone_scores.csv"

MAX_TOKENS = 30  # ~20 words limit

def run_accuracy_test(model, tokenizer):
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "expected_keywords", "model_response", "latency", "accuracy", "model"])
        for q in questions:
            user_q = q["question"]
            expected = q["expected_keywords"]
            prompt = generate_rag_prompt(user_q)
            latency, output_raw = time_inference(model, tokenizer, prompt, max_tokens=MAX_TOKENS)
            cleaned_output = extract_assistant_reply(output_raw, user_q)  # ✅ Trims repetition
            accuracy = evaluate_keywords(cleaned_output, expected)
            writer.writerow([user_q, expected, cleaned_output, round(latency, 2), accuracy, "llama-1b-ft-rag"])

def run_tone_test(model, tokenizer):
    with open(TONE_SAMPLE_PATH) as f:
        samples = json.load(f)

    with open(TONE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "expected_response", "model_response", "similarity", "model"])
        for item in samples:
            prompt = generate_rag_prompt(item["instruction"])
            _, output_raw = time_inference(model, tokenizer, prompt, max_tokens=MAX_TOKENS)
            cleaned_output = extract_assistant_reply(output_raw, item["instruction"])
            similarity = evaluate_tone(cleaned_output, item["response"])
            writer.writerow([item["instruction"], item["response"], cleaned_output, similarity, "llama-1b-ft-rag"])

if __name__ == "__main__":
    tokenizer, model = load_rag_model()
    run_accuracy_test(model, tokenizer)
    run_tone_test(model, tokenizer)
    print("✅ RAG benchmark (accuracy + tone) completed.")
