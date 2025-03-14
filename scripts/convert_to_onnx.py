import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "dheerajdasari/customer-support-1b-4bit"
OUTPUT_ONNX_PATH = "models/customer_support_1b.onnx"

# ✅ Load model in full precision and move it to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto").to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ✅ Create a dummy input and move it to GPU
dummy_input = tokenizer("Hello, how can I assist you?", return_tensors="pt").input_ids.to(device)

# ✅ Export to ONNX (force everything to GPU)
torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_ONNX_PATH,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}},
    opset_version=13
)

print(f"✅ Model successfully converted to ONNX: {OUTPUT_ONNX_PATH}")
