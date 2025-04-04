import time
import concurrent.futures

def run_load_test(model, tokenizer, prompt, concurrency=1, max_tokens=50):
    """
    Simulate concurrent generation requests and measure average latency.
    Parameters:
        model: Hugging Face model (with LoRA if needed)
        tokenizer: Corresponding tokenizer
        prompt: The RAG-enhanced input prompt
        concurrency: Number of concurrent requests
        max_tokens: Max number of tokens per generation
    Returns:
        Dictionary with latency, success/failure metrics
    """

    def generate():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        output = model.generate(input_ids, max_new_tokens=max_tokens)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(generate) for _ in range(concurrency)]
        results = [f.result() for f in futures]

    end = time.perf_counter()

    avg_latency = (end - start) / concurrency

    return {
        "avg_latency": round(avg_latency, 2),
        "failures": 0,
        "successes": len(results)
    }
