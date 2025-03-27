import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
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

# 🔥 Fix for LLaMA: Manually set padding token
tokenizer.pad_token = tokenizer.eos_token

# ✅ Use BitsAndBytesConfig to Speed Up Training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # ✅ Force computations in FP16 for speed
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=MODEL_DIR,
)

# LoRA Configuration (Lightweight Fine-tuning)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # LoRA rank
    lora_alpha=16,
    lora_dropout=0.1,
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# ✅ FIX: Ensure Labels Are Set for Loss Computation
def tokenize_function(examples):
    inputs = tokenizer(
        examples["instruction"], 
        padding="max_length",  
        truncation=True,  
        max_length=512  
    )
    inputs["labels"] = inputs["input_ids"].copy()  # ✅ Add labels for loss computation
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ✅ Training Arguments (Optimized for A3000 6GB)
training_args = TrainingArguments(
    output_dir="./models/llama-finetuned",
    per_device_train_batch_size=2,  # ⬇ Reduce batch size (previously 4)
    gradient_accumulation_steps=2,  # ⬇ Reduce accumulation steps (previously 4)
    num_train_epochs=3,  
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    eval_strategy="no",  # ✅ Updated deprecated argument
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
