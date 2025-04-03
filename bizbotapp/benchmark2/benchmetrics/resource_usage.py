import psutil
import torch

def get_system_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    ram_percent = psutil.virtual_memory().percent
    return {
        "cpu_percent": cpu_percent,
        "ram_percent": ram_percent
    }

def get_gpu_usage():
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1e6  # MB
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e6  # MB
        return {
            "vram_used_MB": round(vram_used, 2),
            "vram_total_MB": round(vram_total, 2),
            "vram_percent": round((vram_used / vram_total) * 100, 2)
        }
    else:
        return {
            "vram_used_MB": 0,
            "vram_total_MB": 0,
            "vram_percent": 0
        }

def get_resource_snapshot():
    sys = get_system_usage()
    gpu = get_gpu_usage()
    return {**sys, **gpu}
