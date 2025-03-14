import os
import torch
import tensorrt as trt
import numpy as np
from transformers import AutoTokenizer
import gradio as gr

# Ensure the models directory exists
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Define TensorRT model path
TRT_MODEL_PATH = "models/customer_support_1b.trt"
TOKENIZER_PATH = "dheerajdasari/customer-support-1b-4bit"  # Use original tokenizer

# Store tokenizer and TensorRT engine
current_tokenizer = None
trt_engine = None

def clear_model():
    """Clears the currently loaded model from memory."""
    global current_tokenizer, trt_engine
    if trt_engine is not None:
        del trt_engine
    if current_tokenizer is not None:
        del current_tokenizer
    torch.cuda.empty_cache()
    current_tokenizer, trt_engine = None, None
    return "Cleared loaded TensorRT model from memory."

def load_tensorrt_model():
    """Loads the TensorRT engine and tokenizer."""
    global trt_engine, current_tokenizer

    clear_model()  # Remove existing model

    # ✅ Load TensorRT engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(TRT_MODEL_PATH, "rb") as f, trt.Runtime(logger) as runtime:
        trt_engine = runtime.deserialize_cuda_engine(f.read())

    # ✅ Load tokenizer
    current_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, cache_dir=MODEL_DIR)

    return "Loaded TensorRT model successfully."

def chat(user_input, instructions, max_tokens, temperature):
    """Processes user input using TensorRT engine."""
    global trt_engine, current_tokenizer

    if trt_engine is None or current_tokenizer is None:
        return "No TensorRT model loaded."

    # ✅ Convert input text to token IDs
    formatted_input = f"{instructions}\nUser: {user_input}\nAssistant:"
    input_ids = current_tokenizer(formatted_input, return_tensors="pt").input_ids.cpu().numpy()

    # ✅ Allocate output buffer
    output = np.empty((1, max_tokens), dtype=np.float32)

    # ✅ Run inference using TensorRT
    with trt_engine.create_execution_context() as context:
        context.execute_v2(bindings=[int(input_ids), int(output)])

    # ✅ Decode output
    response = current_tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return response

# Create Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# TensorRT Customer Support Chatbot")

    load_button = gr.Button("Load TensorRT Model")

    instruction_box = gr.Textbox(label="Instructions for AI (e.g., 'Be brief', 'Use bullet points')")
    user_input = gr.Textbox(label="Enter your question")
    response_box = gr.Textbox(label="Chatbot's Response", interactive=False)

    max_tokens_slider = gr.Slider(label="Response Length (Max Tokens)", minimum=50, maximum=250, value=100)
    temperature_slider = gr.Slider(label="Creativity (Temperature)", minimum=0.1, maximum=1.0, value=0.3)

    submit_button = gr.Button("Submit")
    clear_chat_button = gr.Button("Clear Chat")
    clear_model_button = gr.Button("Clear Loaded Model")

    load_button.click(load_tensorrt_model, outputs=None)
    submit_button.click(chat, inputs=[user_input, instruction_box, max_tokens_slider, temperature_slider], outputs=response_box)
    clear_chat_button.click(lambda: "", outputs=response_box)
    clear_model_button.click(clear_model, outputs=None)

demo.launch()
