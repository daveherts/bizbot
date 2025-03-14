import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import bitsandbytes as bnb

# Ensure model directory exists
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# Model name
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token  # ✅ Fix for LLaMA

# ✅ Use BitsAndBytesConfig for memory-efficient fine-tuning
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # ✅ FP16 for faster training
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=MODEL_DIR,
)

# ✅ Optimized LoRA Config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,  # ⬆ Increase LoRA rank (improves adaptation)
    lora_alpha=64,  # ⬆ Increase alpha for better scaling
    lora_dropout=0.1,
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# ✅ Tokenization (Fix for LLaMA loss masking)
def tokenize_function(examples):
    text = examples["instruction"] + " " + examples["response"]  # ✅ Merge fields for better training
    inputs = tokenizer(
        text, 
        padding="max_length",  
        truncation=True,  
        max_length=512  
    )
    inputs["labels"] = inputs["input_ids"].copy()  # ✅ Add labels for loss computation

    # ✅ Mask padding tokens in loss
    inputs["labels"] = [
        -100 if token == tokenizer.pad_token_id else token for token in inputs["labels"]
    ]

    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ✅ Optimized Training Arguments for A3000 (6GB)
training_args = TrainingArguments(
    output_dir="./models/llama-finetuned",
    per_device_train_batch_size=4,  # ⬆ Increase batch size (if VRAM allows)
    gradient_accumulation_steps=4,  # ⬆ Improve stability
    num_train_epochs=4,  # ⬆ More training for better results
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    evaluation_strategy="epoch",  # ✅ Enable validation
    fp16=True,  # ✅ Mixed Precision for speed
    optim="adamw_bnb_8bit",
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

# 🚀 Start Fine-Tuning
trainer.train()

# Save Model
model.save_pretrained("./models/llama-finetuned")
tokenizer.save_pretrained("./models/llama-finetuned")

print("✅ Training Complete! Model saved in './models/llama-finetuned'")
