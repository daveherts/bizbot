import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# Ensure the models directory exists
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Define available models, including the new customer support model
MODEL_OPTIONS = {
    "Phi-4-Mini-Instruct": "microsoft/Phi-4-mini-instruct",
    "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "LLaMA-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Customer-Support-Instruct-1B": "dheerajdasari/Customer-support-instruct-1B",
    "LLaMA-3.2-3B-Customer-Support (Pretrained)": "kingabzpro/Llama-3.2-3b-it-customer-support",  # ✅ New Model Added
}

# Store the current model to manage memory
current_model = None
current_tokenizer = None

def clear_model():
    """Clears the currently loaded model from memory."""
    global current_model, current_tokenizer
    if current_model is not None:
        del current_model  # Delete model
        del current_tokenizer  # Delete tokenizer
        torch.cuda.empty_cache()  # Free up GPU memory
        current_model, current_tokenizer = None, None  # Reset references
        return "Cleared loaded model from memory."

def load_model(model_name):
    """Loads the selected model and tokenizer, ensuring previous model is removed."""
    global current_model, current_tokenizer
    clear_model()  # Remove any existing model before loading a new one
    
    model_path = MODEL_OPTIONS[model_name]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=MODEL_DIR).to("cuda")

    current_model, current_tokenizer = model, tokenizer
    return f"Loaded {model_name} successfully from {MODEL_DIR}."

def chat(user_input, model_name, instructions, max_tokens, temperature):
    """Processes the user input with the selected model."""
    if model_name not in MODEL_OPTIONS:
        return "Invalid model selection."

    if current_model is None or current_tokenizer is None:
        return "No model loaded. Please select a model first."

    # ✅ Improved input formatting for better response generation
    formatted_input = f"{instructions}\nUser: {user_input}\nAssistant:"

    inputs = current_tokenizer(formatted_input, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = current_model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=temperature, pad_token_id=current_tokenizer.eos_token_id
        )

    response = current_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # ✅ Clean up unnecessary system/user parts
    response = response.replace("User:", "").replace("Assistant:", "").strip()
    
    return response

# Create Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Conversational AI Chatbot")

    model_dropdown = gr.Dropdown(label="Select Model", choices=list(MODEL_OPTIONS.keys()), value="LLaMA-3.2-3B-Customer-Support (Pretrained)")  # ✅ Default to new customer support model
    load_button = gr.Button("Load Model")

    instruction_box = gr.Textbox(label="Instructions for AI (e.g., 'Be brief', 'Use bullet points')")
    user_input = gr.Textbox(label="Enter your question")
    
    response_box = gr.Textbox(label="Chatbot's Response", interactive=False)
    
    max_tokens_slider = gr.Slider(label="Response Length (Max Tokens)", minimum=50, maximum=250, value=100)  # ✅ Reduced max tokens
    temperature_slider = gr.Slider(label="Creativity (Temperature)", minimum=0.1, maximum=1.0, value=0.3)  # ✅ Lowered temperature for structured responses
    
    submit_button = gr.Button("Submit")
    clear_chat_button = gr.Button("Clear Chat")
    clear_model_button = gr.Button("Clear Loaded Model")

    load_button.click(load_model, inputs=[model_dropdown], outputs=None)
    submit_button.click(chat, inputs=[user_input, model_dropdown, instruction_box, max_tokens_slider, temperature_slider], outputs=response_box)
    clear_chat_button.click(lambda: "", outputs=response_box)
    clear_model_button.click(clear_model, outputs=None)

demo.launch()
