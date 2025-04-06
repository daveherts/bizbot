import os
import json
import csv
import time
import statistics
import concurrent.futures
import random
import psutil

from rag.rag.bot import BizBot
from benchmark2.benchmetrics.accuracy import evaluate_keywords
from benchmark2.utils import extract_assistant_reply

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_ENABLED = True
except Exception:
    GPU_ENABLED = False

bot = BizBot()

QUESTIONS_PATH = "rag/mixed_benchmark_questions.json"
OUTPUT_DIR = "rag/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_resource_snapshot():
    cpu_percent = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    mem_used = ram.used / (1024 ** 3)
    mem_total = ram.total / (1024 ** 3)

    gpu_usage = None
    gpu_mem_used = None
    if GPU_ENABLED:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem_used = mem_info.used / (1024 ** 3)

    return {
        "cpu_percent": round(cpu_percent, 2),
        "ram_used_gb": round(mem_used, 2),
        "ram_total_gb": round(mem_total, 2),
        "gpu_percent": gpu_usage,
        "gpu_memory_used_gb": gpu_mem_used
    }

def test_question_under_load(question_data):
    user_q = question_data["question"]
    expected = question_data["expected_keywords"]

    start = time.time()
    raw_response = bot.answer(user_q)
    latency = round(time.time() - start, 3)

    cleaned = extract_assistant_reply(raw_response, user_q)
    accuracy = evaluate_keywords(cleaned, expected)

    return {
        "question": user_q,
        "expected_keywords": expected,
        "model_response": cleaned,
        "latency": latency,
        "accuracy": accuracy
    }

def run_mixed_concurrent_test(concurrent_users: int, output_name: str):
    with open(QUESTIONS_PATH) as f:
        all_questions = json.load(f)

    # Split into FAQ vs Generated
    faq_qs = [q for q in all_questions if q["expected_keywords"]]
    gen_qs = [q for q in all_questions if not q["expected_keywords"]]

    # Sample 50/50 for the given load
    faq_sample = random.choices(faq_qs, k=concurrent_users // 2)
    gen_sample = random.choices(gen_qs, k=concurrent_users - len(faq_sample))
    test_pool = faq_sample + gen_sample
    random.shuffle(test_pool)

    print(f"\n🚀 Load test with {concurrent_users} users (FAQ={len(faq_sample)} / Gen={len(gen_sample)})")

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        results = list(executor.map(test_question_under_load, test_pool))

    latencies = [r["latency"] for r in results]
    resource = get_resource_snapshot()

    output_file = os.path.join(OUTPUT_DIR, output_name + ".csv")
    summary_file = os.path.join(OUTPUT_DIR, output_name + "_summary.csv")

    with open(output_file, "w", newline="") as f:
        fieldnames = ["question", "expected_keywords", "model_response", "latency", "accuracy", "model"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            r["model"] = "bizbot-llama1b-ft-rag"
            writer.writerow(r)

    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concurrency", "avg_latency", "p95_latency", "max_latency", "cpu_percent", "ram_used_gb", "gpu_percent", "gpu_memory_used_gb"])
        writer.writerow([
            concurrent_users,
            round(statistics.mean(latencies), 2),
            round(statistics.quantiles(latencies, n=100)[94], 2),
            round(max(latencies), 2),
            resource.get("cpu_percent"),
            resource.get("ram_used_gb"),
            resource.get("gpu_percent"),
            resource.get("gpu_memory_used_gb")
        ])

    print(f"✅ Saved to {output_file} and {summary_file}")

if __name__ == "__main__":
    run_mixed_concurrent_test(10, "mixed_loadtest_10")
    run_mixed_concurrent_test(20, "mixed_loadtest_20")
    run_mixed_concurrent_test(50, "mixed_loadtest_50")
