import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import onnx
import onnxruntime

# Define model directory
model_dir = "/home/dave/projects/bizbot/models/models--unsloth--llama-3.2-3b-instruct-bnb-4bit"

# Check if model directory exists
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory not found: {model_dir}")

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
except Exception as e:
    raise RuntimeError(f"Failed to load tokenizer: {e}")

# Load model
try:
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()  # Set model to evaluation mode
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Define ONNX export path
onnx_model_path = os.path.join(model_dir, "model.onnx")

# Define dummy input
dummy_input = tokenizer("Convert this to ONNX", return_tensors="pt").input_ids

# Export the model to ONNX
try:
    torch.onnx.export(
        model, dummy_input, onnx_model_path,
        input_names=["input_ids"], output_names=["output"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"},
                      "output": {0: "batch_size", 1: "sequence_length"}},
        opset_version=12
    )
    print(f"ONNX model saved at {onnx_model_path}")
except Exception as e:
    raise RuntimeError(f"Failed to convert to ONNX: {e}")

# Verify the exported ONNX model
try:
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")
except Exception as e:
    raise RuntimeError(f"ONNX model validation failed: {e}")

# Test inference with ONNX Runtime
try:
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    ort_inputs = {"input_ids": dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    print("ONNX model inference successful!")
except Exception as e:
    raise RuntimeError(f"ONNX inference failed: {e}")
