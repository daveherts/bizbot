import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

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

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    return tokenizer, model
