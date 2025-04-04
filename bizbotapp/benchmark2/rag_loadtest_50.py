import csv
from .ragmodel import load_rag_model, generate_rag_prompt
from .benchmetrics.load_test import run_load_test
from .benchmetrics.resource_usage import get_resource_snapshot

# === Config ===
PROMPT = "What allergens are in the Almond Mocha Ground?"
CONCURRENCY = 50
MAX_TOKENS = 30  # Limit to ~20 words
RESULTS_CSV = "benchmark2/results/rag_loadtest_50.csv"

def main():
    # Load fine-tuned LLaMA model + adapter
    tokenizer, model = load_rag_model()

    # Construct RAG-enhanced prompt using vector search
    prompt = generate_rag_prompt(PROMPT)

    print(f"\n🚀 Running RAG load test @ {CONCURRENCY} users with {MAX_TOKENS} tokens")

    # Run the concurrent generation test
    results = run_load_test(model, tokenizer, prompt, concurrency=CONCURRENCY, max_tokens=MAX_TOKENS)

    # Collect system resource stats
    resources = get_resource_snapshot()
    results.update(resources)

    # Append identifying metadata
    results["model"] = "llama-1b-ft-rag"
    results["concurrency"] = CONCURRENCY
    results["max_tokens"] = MAX_TOKENS

    # Write results to CSV
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)

    print(f"\n✅ Results saved to {RESULTS_CSV}")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
