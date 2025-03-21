import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

MODEL_DIR = "/home/dave/project/bizbot/bizbot/models"
BASE_MODEL_PATH = f"{MODEL_DIR}/Llama-3.2-1B-Instruct"
ADAPTER_PATH = f"{MODEL_DIR}/llamaft/checkpoint-20154"

device = "cuda" if torch.cuda.is_available() else "cpu"

current_model = None
current_tokenizer = None
conversation_history = []

# Embedded instructions and parameters
INSTRUCTIONS = "you are a customer service chat bot"
MAX_TOKENS = 50
TEMPERATURE = 0.22

def ensure_base_model():
    if not os.path.exists(BASE_MODEL_PATH):
        print("Base model not found. Downloading...")
        os.makedirs(BASE_MODEL_PATH, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.float16)
        tokenizer.save_pretrained(BASE_MODEL_PATH)
        model.save_pretrained(BASE_MODEL_PATH)
        print("Base model downloaded successfully.")
    else:
        print("Base model already exists locally.")

def clear_model():
    global current_model, current_tokenizer
    if current_model:
        del current_model
        del current_tokenizer
        torch.cuda.empty_cache()
        current_model, current_tokenizer = None, None
        return "Cleared loaded model from memory."

def load_bizbot():
    global current_model, current_tokenizer
    clear_model()
    ensure_base_model()

    current_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    current_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        local_files_only=True,
        torch_dtype=torch.float16
    ).to(device)

    return "✅ BizBot fine-tuned model loaded successfully."

def chat(user_input):
    global conversation_history

    if current_model is None or current_tokenizer is None:
        return "Model not loaded, please reload."

    conversation_history.append(f"User: {user_input}")
    formatted_input = (
        f"System: You are an AI assistant. Follow user instructions carefully.\n"
        f"Instructions: {INSTRUCTIONS}\n"
        + "\n".join(conversation_history)
        + "\nAssistant:"
    )

    inputs = current_tokenizer(formatted_input, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = current_model.generate(
            **inputs, max_new_tokens=MAX_TOKENS, temperature=TEMPERATURE
        )

    response = current_tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Assistant:")[-1].strip()

    conversation_history.append(f"Assistant: {response}")

    return response

# Automatically load BizBot at startup
load_bizbot()

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 BizBot AI Chatbot")

    user_input = gr.Textbox(label="Enter your question")
    response_box = gr.Textbox(label="Chatbot's Response", interactive=False)

    submit_button = gr.Button("Submit")
    clear_chat_button = gr.Button("Clear Chat")

    submit_button.click(chat, inputs=user_input, outputs=response_box)
    clear_chat_button.click(lambda: "", outputs=response_box)

demo.launch()