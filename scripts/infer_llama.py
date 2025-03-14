import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model
MODEL_PATH = "./models/llama-finetuned"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cuda")

def chat(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test the fine-tuned model
test_prompt = "How do I cancel my order?"
print("🤖 AI Response:", chat(test_prompt))
