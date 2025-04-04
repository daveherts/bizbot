import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Mapping for your fine-tuned models
MODEL_MAP = {
    "fine-tuned-llama-1b": {
        "base": "./models/base/Llama-3.2-1B-Instruct",
        "adapter": "./models/adapters/llamaft/checkpoint-20154"
    }
}

def load_model(model_key):
    entry = MODEL_MAP.get(model_key)

    if not entry or not isinstance(entry, dict):
        raise ValueError(f"Unknown fine-tuned model key: {model_key}")

    base_path = entry["base"]
    adapter_path = entry["adapter"]

    # Load tokenizer from base model directory
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    # Load base model with accelerate-style device mapping
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map="auto"  # This dispatches across available GPU/CPU memory
    )

    # Load LoRA adapter on top of base
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.float16
    )

    return tokenizer, model
