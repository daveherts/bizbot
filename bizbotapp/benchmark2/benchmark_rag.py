import os
import json
import csv
import time
from rag.rag.bot import BizBot
from rag.rag.prompt_template import format_prompt
from benchmark2.benchmetrics.accuracy import evaluate_keywords
from benchmark2.benchmetrics.tone_analysis import evaluate_tone
from benchmark2.utils import extract_assistant_reply

# File paths
QUESTIONS_PATH = "benchmark2/data/benchmark_questions.json"
TONE_SAMPLE_PATH = "benchmark2/data/bitext_sample.json"
RESULTS_CSV = "benchmark2/results/rag_full_results.csv"
TONE_CSV = "benchmark2/results/rag_tone_scores.csv"

# Load RAG bot
bot = BizBot()

# --- Prompt Generator (no system_instructions arg now) ---
def generate_rag_prompt(query: str) -> str:
    return format_prompt(query, context="")  

# --- Accuracy Test ---
def run_accuracy_test():
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "expected_keywords", "model_response", "latency", "accuracy", "model"])

        for q in questions:
            user_q = q["question"]
            expected = q["expected_keywords"]

            start = time.time()
            output_raw = bot.answer(user_q)
            latency = round(time.time() - start, 2)

            cleaned_output = extract_assistant_reply(output_raw, user_q)
            accuracy = evaluate_keywords(cleaned_output, expected)

            writer.writerow([user_q, expected, cleaned_output, latency, accuracy, "bizbot-llama1b-ft-rag"])
            print(f"{user_q} | ⏱ {latency}s | 🎯 {accuracy*100:.1f}%")

# --- Tone Test ---
def run_tone_test():
    with open(TONE_SAMPLE_PATH) as f:
        samples = json.load(f)

    with open(TONE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "expected_response", "model_response", "similarity", "model"])

        for item in samples:
            prompt = item["instruction"]

            response_raw = bot.answer(prompt)
            cleaned_output = extract_assistant_reply(response_raw, prompt)
            similarity = evaluate_tone(cleaned_output, item["response"])

            writer.writerow([prompt, item["response"], cleaned_output, similarity, "bizbot-llama1b-ft-rag"])
            print(f"🎤 Tone for: {prompt[:40]}... | 🔗 Sim: {similarity:.2f}")

# --- Run All ---
if __name__ == "__main__":
    run_accuracy_test()
    run_tone_test()
    print("RAG benchmark complete.")
