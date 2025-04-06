import os
import json
import csv
import time
import statistics
import concurrent.futures
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_ENABLED = True
except Exception:
    GPU_ENABLED = False

from rag.bot import BizBot
from benchmark2.benchmetrics.accuracy import evaluate_keywords
from benchmark2.utils import extract_assistant_reply

# === Paths ===
QUESTIONS_PATH = os.path.abspath("benchmark2/data/benchmark_questions.json")
OUTPUT_DIR = os.path.abspath("rag/results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Shared Bot Instance ===
bot = BizBot()

def get_resource_snapshot():
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    gpu_usage = "n/a"
    gpu_mem = "n/a"

    if GPU_ENABLED:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem = mem_info.used / (1024 ** 3)
        except Exception:
            pass

    return {
        "cpu_percent": round(cpu, 2),
        "ram_used_gb": round(ram.used / (1024 ** 3), 2),
        "gpu_percent": gpu_usage,
        "gpu_memory_used_gb": gpu_mem
    }

def run_test(question):
    q = question["question"]
    expected = question["expected_keywords"]
    start = time.time()
    response = bot.answer(q)
    latency = round(time.time() - start, 3)
    clean = extract_assistant_reply(response, q)
    accuracy = evaluate_keywords(clean, expected)
    return {
        "question": q,
        "expected_keywords": expected,
        "model_response": clean,
        "latency": latency,
        "accuracy": accuracy
    }

def run_concurrent(concurrency: int):
    print(f"\n🚀 Running benchmark for {concurrency} users...")
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    pool = questions * ((concurrency + len(questions) - 1) // len(questions))
    pool = pool[:concurrency]

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(run_test, pool))

    latencies = [r["latency"] for r in results]
    metrics = get_resource_snapshot()

    csv_name = f"faq_loadtest_{concurrency}.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_name)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "expected_keywords", "model_response", "latency", "accuracy", "model"])
        for r in results:
            writer.writerow([r["question"], r["expected_keywords"], r["model_response"], r["latency"], r["accuracy"], "bizbot-faq"])

    # Save summary
    summary_name = csv_name.replace(".csv", "_summary.csv")
    with open(os.path.join(OUTPUT_DIR, summary_name), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concurrency", "avg_latency", "p95_latency", "max_latency", "cpu_percent", "ram_used_gb", "gpu_percent", "gpu_memory_used_gb"])
        writer.writerow([
            concurrency,
            round(statistics.mean(latencies), 2),
            round(statistics.quantiles(latencies, n=100)[94], 2),
            round(max(latencies), 2),
            metrics["cpu_percent"],
            metrics["ram_used_gb"],
            metrics["gpu_percent"],
            metrics["gpu_memory_used_gb"]
        ])

    print(f"✅ Done. Saved to {csv_path}")

if __name__ == "__main__":
    run_concurrent(10)
    run_concurrent(20)
    run_concurrent(50)
    print("\n🏁 All done!")
