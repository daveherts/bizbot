import os
import json
import csv
import time
import statistics
import threading
import concurrent.futures
import psutil

from rag.rag.bot import BizBot
from benchmark2.benchmetrics.accuracy import evaluate_keywords
from benchmark2.utils import extract_assistant_reply

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_ENABLED = True
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    GPU_ENABLED = False
    GPU_HANDLE = None

# Load bot once globally
bot = BizBot()

# Constants
QUESTIONS_PATH = "benchmark2/data/benchmark_questions.json"
OUTPUT_DIR = "benchmark2/results/load_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

gpu_stats = []

def sample_gpu_usage(interval=0.5):
    while getattr(threading.current_thread(), "keep_running", True):
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE).used / (1024**3)
            gpu_stats.append((util, mem))
        except Exception:
            pass
        time.sleep(interval)

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

    print(f"\n Starting load test with {concurrent_users} users...")

    # Start GPU sampling in a background thread
    gpu_thread = threading.Thread(target=sample_gpu_usage)
    gpu_thread.keep_running = True
    if GPU_ENABLED:
        gpu_thread.start()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        results = list(executor.map(test_question_under_load, test_pool))

    if GPU_ENABLED:
        gpu_thread.keep_running = False
        gpu_thread.join()

    latencies = [r["latency"] for r in results]
    cpu_percent = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()

    avg_gpu = round(statistics.mean([x[0] for x in gpu_stats]), 2) if gpu_stats else None
    peak_gpu_mem = round(max([x[1] for x in gpu_stats]), 2) if gpu_stats else None

    # Write detailed results
    with open(output_file, "w", newline="") as f:
        fieldnames = ["question", "expected_keywords", "model_response", "latency", "accuracy", "model"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            r["model"] = "bizbot-llama1b-ft-rag"
            writer.writerow(r)

    # Write summary file
    summary_path = output_file.replace(".csv", "_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concurrency", "avg_latency", "p95_latency", "max_latency",
                         "cpu_percent", "ram_used_gb", "gpu_percent_avg", "gpu_mem_peak_gb"])
        writer.writerow([
            concurrent_users,
            round(statistics.mean(latencies), 2),
            round(statistics.quantiles(latencies, n=100)[94], 2),
            round(max(latencies), 2),
            round(cpu_percent, 2),
            round(ram.used / (1024**3), 2),
            avg_gpu,
            peak_gpu_mem
        ])

    print(f" Results saved to {output_file} and {summary_path}")

if __name__ == "__main__":
    run_concurrent_test(10, os.path.join(OUTPUT_DIR, "loadtest_10_users.csv"))
    run_concurrent_test(20, os.path.join(OUTPUT_DIR, "loadtest_20_users.csv"))
    run_concurrent_test(50, os.path.join(OUTPUT_DIR, "loadtest_50_users.csv"))
