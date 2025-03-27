import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

BASE_MODEL_PATH = "/home/dave/project/bizbot/bizbot/models/models--meta-llama--Llama-3.2-1B-Instruct"
ADAPTER_PATH = "/home/dave/project/bizbot/bizbot/models/llamaft/checkpoint-20154"

device = "cuda" if torch.cuda.is_available() else "cpu"

current_model = None
current_tokenizer = None

def clear_model():
    global current_model, current_tokenizer
    if current_model:
        del current_model
        del current_tokenizer
        torch.cuda.empty_cache()
        current_model, current_tokenizer = None, None
        return "Model cleared from memory."

def load_model():
    global current_model, current_tokenizer
    clear_model()

    current_tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load your PEFT adapters explicitly
    current_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        local_files_only=True,
        torch_dtype=torch.float16
    ).to(device)

    return "✅ BizBot PEFT model loaded successfully."

def chat(user_input, instructions, max_tokens, temperature):
    if current_model is None or current_tokenizer is None:
        return "Please load the model first."

    formatted_input = (
        f"System: You are an AI assistant. Follow user instructions carefully.\n"
        f"Instructions: {instructions}\n"
        f"User: {user_input}\n"
        f"Assistant:"
    )

    inputs = current_tokenizer(formatted_input, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = current_model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=temperature
        )

    response = current_tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace("User:", "").replace("Assistant:", "").strip()

    return response

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 BizBot AI Chatbot (PEFT Adapter)")

    load_button = gr.Button("🚀 Load BizBot Model")
    instruction_box = gr.Textbox(label="Instructions for AI (e.g., 'Be brief', 'Use bullet points')")
    user_input = gr.Textbox(label="Enter your question")
    response_box = gr.Textbox(label="Chatbot's Response", interactive=False)

    max_tokens_slider = gr.Slider(50, 500, value=150, label="Response Length (Max Tokens)")
    temperature_slider = gr.Slider(0.1, 1.5, value=0.7, label="Creativity (Temperature)")

    submit_button = gr.Button("Submit")
    clear_chat_button = gr.Button("Clear Chat")
    clear_model_button = gr.Button("Clear Loaded Model")

    load_button.click(load_model, outputs=None)
    submit_button.click(
        chat,
        inputs=[user_input, instruction_box, max_tokens_slider, temperature_slider],
        outputs=response_box
    )
    clear_chat_button.click(lambda: "", outputs=response_box)
    clear_model_button.click(clear_model, outputs=None)

demo.launch()
