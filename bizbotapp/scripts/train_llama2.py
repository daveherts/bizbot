import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# ✅ Ensure model directory exists
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ Load dataset (Customer Support Chatbot Dataset)
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# ✅ Model & Tokenizer (No bitsandbytes)
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)

# 🔥 Fix for LLaMA: Set padding token
tokenizer.pad_token = tokenizer.eos_token

# ✅ Load model (NO 4-bit Quantization, uses FP16 for TensorRT compatibility)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # ✅ Use FP16 instead of 4-bit
    device_map="auto",
    cache_dir=MODEL_DIR,
)

# ✅ LoRA Configuration (for Efficient Fine-Tuning)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # LoRA rank (balance memory & performance)
    lora_alpha=16,
    lora_dropout=0.1,
)

# ✅ Apply LoRA to model
model = get_peft_model(model, lora_config)

# ✅ Tokenization Function (Fixed for Batched Input)
def tokenize_function(examples):
    """
    Processes a batch of examples by concatenating instructions and responses
    """
    text = [inst + " " + resp for inst, resp in zip(examples["instruction"], examples["response"])]

    inputs = tokenizer(
        text,
        padding="max_length",  # 🚀 Ensures equal-length batches
        truncation=True,
        max_length=512,  # 🔥 Adjust based on VRAM
    )
    inputs["labels"] = inputs["input_ids"].copy()  # ✅ Ensure labels for loss computation
    return inputs

# ✅ Preprocess Dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# ✅ Training Arguments (Optimized for A3000 6GB VRAM)
training_args = TrainingArguments(
    output_dir="./models/llama-finetuned",
    per_device_train_batch_size=2,  # 🔥 Reduce batch size for low VRAM
    gradient_accumulation_steps=2,  # 🔥 Accumulate gradients for better optimization
    num_train_epochs=3,  
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    evaluation_strategy="no",  # ✅ No evaluation (as dataset may not have validation set)
    fp16=True,  # ✅ Mixed Precision for efficiency (Compatible with TensorRT)
    optim="adamw_torch",  # ✅ Use standard AdamW (No bitsandbytes)
    report_to="none",
)

# ✅ Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

# 🚀 **Start Fine-Tuning**
trainer.train()

# ✅ Save Model (Prepare for ONNX Conversion)
model.save_pretrained("./models/llama-finetuned")
tokenizer.save_pretrained("./models/llama-finetuned")

print("✅ Training Complete! Model saved in './models/llama-finetuned'")
