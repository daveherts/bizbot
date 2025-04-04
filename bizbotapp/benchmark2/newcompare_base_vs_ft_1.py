import json
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from benchmetrics.load_test import run_load_test
from benchmetrics.resource_usage import get_resource_snapshot
from utils import clean_response

# Model paths
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
FT_BASE_PATH = "./models/base/Llama-3.2-1B-Instruct"
FT_ADAPTER_PATH = "./models/adapters/llamaft/checkpoint-20154"

# Benchmark config
QUESTIONS_PATH = "benchmark2/data/benchmark_questions.json"
RESULTS_CSV = "benchmark2/results/newcompare_base_vs_ft_1.csv"
MAX_TOKENS = 30
CONCURRENCY = 1

def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype="auto"
    )
    return tokenizer, model

def load_finetuned_model():
    tokenizer = AutoTokenizer.from_pretrained(FT_BASE_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        FT_BASE_PATH,
        device_map="auto",
        torch_dtype="auto"
    )
    model = PeftModel.from_pretrained(
        base_model,
        FT_ADAPTER_PATH,
        torch_dtype="auto"
    )
    return tokenizer, model

def benchmark_model(model_key, tokenizer, model, questions):
    rows = []
    print(f"\n🚀 Running load test for: {model_key} @ {CONCURRENCY} users")

    for q in questions:
        prompt = q["question"]
        result = run_load_test(model, tokenizer, prompt, concurrency=CONCURRENCY, max_tokens=MAX_TOKENS)
        resources = get_resource_snapshot()
        row = {
            **result,
            **resources,
            "model": model_key,
            "question": prompt,
            "max_tokens": MAX_TOKENS
        }
        rows.append(row)

    return rows

def save_results(rows):
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def main():
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    results = []

    base_tokenizer, base_model = load_base_model()
    results.extend(benchmark_model("llama-1b-base-short", base_tokenizer, base_model, questions))

    ft_tokenizer, ft_model = load_finetuned_model()
    results.extend(benchmark_model("llama-1b-ft-short", ft_tokenizer, ft_model, questions))

    save_results(results)
    print(f"\n✅ Results saved to {RESULTS_CSV}")

if __name__ == "__main__":
    main()
