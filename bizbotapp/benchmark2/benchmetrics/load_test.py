from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import torch

def generate_response(model, tokenizer, prompt):
    device = model.device
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.2)
        latency = time.time() - start
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return latency, response, False  # No failure
    except Exception as e:
        return None, str(e), True  # Failure

def run_load_test(model, tokenizer, prompt, concurrency=10):
    latencies = []
    failures = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(generate_response, model, tokenizer, prompt) for _ in range(concurrency)]
        for future in as_completed(futures):
            latency, _, failed = future.result()
            if failed:
                failures += 1
            elif latency is not None:
                latencies.append(latency)

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
    else:
        avg_latency = None

    return {
        "avg_latency": round(avg_latency, 2) if avg_latency else None,
        "failures": failures,
        "successes": len(latencies),
        "concurrency": concurrency
    }
