import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local directory to store downloaded models
HF_MODEL_DIR = os.path.join(os.path.dirname(__file__), "hf_models")
os.makedirs(HF_MODEL_DIR, exist_ok=True)

# Model keys and corresponding Hugging Face Hub IDs
MODEL_MAP = {
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "phi-2": "microsoft/phi-2",
    "phi-4-mini": "microsoft/Phi-4-mini-instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "gemma-1b": "google/gemma-3-1b-it",
    "gemma-7b": "google/gemma-7b"
}

def load_model(model_key):
    if model_key not in MODEL_MAP:
        raise ValueError(f"Unknown model key: {model_key}")

    model_id = MODEL_MAP[model_key]
    local_dir = os.path.join(HF_MODEL_DIR, model_key)

    print(f"✅ Loading model: {model_id}")
    
    # Use Hugging Face's built-in cache mechanism
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=local_dir)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=local_dir, torch_dtype=torch.float16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return tokenizer, model
