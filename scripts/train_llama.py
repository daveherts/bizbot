import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import bitsandbytes as bnb

# Ensure model directory exists
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset from Hugging Face
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# Load model and tokenizer with quantization
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,  # Load in 4-bit precision (QLoRA)
    device_map="auto",
    cache_dir=MODEL_DIR,
)

# LoRA Configuration (Lightweight Fine-tuning)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Rank of LoRA
    lora_alpha=16,
    lora_dropout=0.1,
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Preprocessing: Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["instruction"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./models/llama-finetuned",
    per_device_train_batch_size=4,  # Small batch size due to 6GB VRAM
    gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
    num_train_epochs=3,  # Adjust based on performance
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    fp16=True,  # Use mixed precision for speed
    optim="adamw_bnb_8bit",  # Memory-efficient optimizer
    report_to="none",  # Disable logging to Hugging Face
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

# Fine-Tune the Model
trainer.train()

# Save Model
model.save_pretrained("./models/llama-finetuned")
tokenizer.save_pretrained("./models/llama-finetuned")

print("✅ Training Complete! Model saved in './models/llama-finetuned'")
