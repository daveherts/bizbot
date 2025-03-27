import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define paths
model_dir = "/home/dave/projects/bizbot/models/models--unsloth--llama-3.2-3b-instruct-bnb-4bit"
onnx_model_path = "/home/dave/projects/bizbot/models/llama-3.2-3b-instruct.onnx"

# Force CPU usage for ONNX export
device = torch.device("cpu")
print(f"Using device: {device}")

# Ensure model directory exists
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory not found: {model_dir}")

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    print("✅ Tokenizer loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load tokenizer: {e}")

# Load model WITHOUT 4-bit quantization
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,  # Load in FP32 (or use torch.float16 for FP16)
        load_in_4bit=False,         # Ensure no 4-bit quantization
        local_files_only=True
    )
    model.to(device)  # Move model to CPU for ONNX export
    print("✅ Model loaded successfully in FP32!")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# Function to export model to ONNX
def export_to_onnx(model, tokenizer, onnx_path):
    dummy_input = tokenizer("Hello, how are you?", return_tensors="pt").input_ids.to(device)

    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}},
        opset_version=13
    )
    print(f"✅ ONNX model exported to: {onnx_path}")

# Export to ONNX
export_to_onnx(model, tokenizer, onnx_model_path)
print("🎉 ONNX export completed successfully!")
