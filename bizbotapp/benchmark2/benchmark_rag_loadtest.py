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

from rag.rag.bot import BizBot
from benchmark2.benchmetrics.accuracy import evaluate_keywords
from benchmark2.utils import extract_assistant_reply

# Load bot once globally
bot = BizBot()

# Constants
QUESTIONS_PATH = "benchmark2/data/benchmark_questions.json"
OUTPUT_DIR = "benchmark2/results/load_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_resource_snapshot():
    cpu_percent = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    memory_used = ram.used / (1024 ** 3)
    memory_total = ram.total / (1024 ** 3)

    gpu_usage = "N/A"
    gpu_memory_used = "N/A"
    if GPU_ENABLED:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_usage = util.gpu
            gpu_memory_used = round(mem.used / (1024 ** 3), 2)
            print(f"📈 GPU Util: {gpu_usage}% | GPU Mem: {gpu_memory_used} GB")
        except Exception as e:
            print(f"⚠️ GPU read error: {e}")

    return {
        "cpu_percent": round(cpu_percent, 2),
        "ram_used_gb": round(memory_used, 2),
        "ram_total_gb": round(memory_total, 2),
        "gpu_percent": gpu_usage,
        "gpu_memory_used_gb": gpu_memory_used
    }

def test_question_under_load(question_data):
    user_q = question_data["question"]
    expected = question_data["expected_keywords"]

    start = time.time()
    raw_response = bot.answer(user_q)
    latency = round(time.time() - start, 3)

    cleaned_output = extract_assistant_reply(raw_response, user_q)
    accuracy = evaluate_keywords(cleaned_output, expected)

    return {
        "question": user_q,
        "expected_keywords": expected,
        "model_response": cleaned_output,
        "latency": latency,
        "accuracy": accuracy
    }

def run_concurrent_test(concurrent_users: int, output_file: str):
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    test_pool = questions * ((concurrent_users + len(questions) - 1) // len(questions))
    test_pool = test_pool[:concurrent_users]

    print(f"\n🚀 Starting load test with {concurrent_users} users...")

    if GPU_ENABLED:
        pynvml.nvmlShutdown()
        pynvml.nvmlInit()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        results = list(executor.map(test_question_under_load, test_pool))

    latencies = [r["latency"] for r in results]
    resource = get_resource_snapshot()

    print(f"📊 Latency Summary ({concurrent_users} users):")
    print(f"   ⏱ Avg: {round(statistics.mean(latencies), 2)}s")
    print(f"   🔺 95th percentile: {round(statistics.quantiles(latencies, n=100)[94], 2)}s")
    print(f"   🚨 Max: {round(max(latencies), 2)}s")

    with open(output_file, "w", newline="") as f:
        fieldnames = ["question", "expected_keywords", "model_response", "latency", "accuracy", "model"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            r["model"] = "bizbot-llama1b-ft-rag"
            writer.writerow(r)

    summary_path = output_file.replace(".csv", "_summary.csv")
    with open(summary_path, "w", newline="") as f:
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
            resource.get("gpu_memory_used_gb"),
        ])

    print(f"✅ Results saved to {output_file} and {summary_path}")

if __name__ == "__main__":
    run_concurrent_test(10, os.path.join(OUTPUT_DIR, "loadtest_10_users.csv"))
    run_concurrent_test(20, os.path.join(OUTPUT_DIR, "loadtest_20_users.csv"))
    run_concurrent_test(50, os.path.join(OUTPUT_DIR, "loadtest_50_users.csv"))
