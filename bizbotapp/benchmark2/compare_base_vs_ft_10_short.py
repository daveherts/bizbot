import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from benchmetrics.load_test import run_load_test
from benchmetrics.resource_usage import get_resource_snapshot

# Model paths
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
FT_BASE_PATH = "./models/base/Llama-3.2-1B-Instruct"
FT_ADAPTER_PATH = "./models/adapters/llamaft/checkpoint-20154"

RESULTS_CSV = "benchmark2/results/base_vs_ft_loadtest_10_short.csv"
PROMPT = "How do I cancel my subscription?"
CONCURRENCY = 10
MAX_TOKENS = 30  # ~20 words

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
        torch_dtype="auto",
        device_map="auto"
    )
    model = PeftModel.from_pretrained(
        base_model,
        FT_ADAPTER_PATH,
        torch_dtype="auto"
    )
    return tokenizer, model

def benchmark_model(model_key, tokenizer, model):
    print(f"\n🚀 Running load test for: {model_key} @ {CONCURRENCY} users with {MAX_TOKENS} max tokens")
    load_results = run_load_test(
        model=model,
        tokenizer=tokenizer,
        prompt=PROMPT,
        concurrency=CONCURRENCY
    )
    resources = get_resource_snapshot()
    combined = {**load_results, **resources}
    combined["model"] = model_key
    combined["max_tokens"] = MAX_TOKENS
    return combined

def save_results(rows):
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def main():
    results = []

    base_tokenizer, base_model = load_base_model()
    results.append(benchmark_model("llama-1b-base-short", base_tokenizer, base_model))

    ft_tokenizer, ft_model = load_finetuned_model()
    results.append(benchmark_model("llama-1b-ft-short", ft_tokenizer, ft_model))

    save_results(results)
    print(f"\n✅ Results saved to {RESULTS_CSV}")

if __name__ == "__main__":
    main()
