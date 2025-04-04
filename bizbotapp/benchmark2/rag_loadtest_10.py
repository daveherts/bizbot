import csv
from ragmodel import load_rag_model, generate_rag_prompt
from benchmetrics.load_test import run_load_test
from benchmetrics.resource_usage import get_resource_snapshot

PROMPT = "What allergens are in the Almond Mocha Ground?"
CONCURRENCY = 10
MAX_TOKENS = 30  # Limit to ~20 words
RESULTS_CSV = "benchmark2/results/rag_loadtest_10.csv"

def main():
    tokenizer, model = load_rag_model()
    prompt = generate_rag_prompt(PROMPT)

    print(f"\n🚀 Running RAG load test @ {CONCURRENCY} users with {MAX_TOKENS} tokens")
    results = run_load_test(model, tokenizer, prompt, concurrency=CONCURRENCY, max_tokens=MAX_TOKENS)
    resources = get_resource_snapshot()
    results.update(resources)
    results["model"] = "llama-1b-ft-rag"
    results["concurrency"] = CONCURRENCY
    results["max_tokens"] = MAX_TOKENS

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)

    print(f"\n✅ Results saved to {RESULTS_CSV}")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
