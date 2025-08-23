# train.py
# Description
# This script runs the complete fine-tuning process for the SightGuide model
# using the prepared VizWiz dataset.
# Author: Mazhar
# Created on: Aug 23, 2025

from math import e
from librosa import ex
import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from datasets import load_from_disk
from trl import SFTTrainer
import os
import yaml
import sys
from utils import make_clean_dir

# Define the path to the configuration file
CONFIG_FILE_PATH = "fine-tune/configs/configs.yaml"

try:
    # --- Load Configuration ---
    with open(CONFIG_FILE_PATH, "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f"❌ ERROR: Failed to load configuration file '{CONFIG_FILE_PATH}': {e}")
    sys.exit(1)

# --- Step 1: Configuration ---
# Define constants and configurations at the top for easy access.
MODEL_ID = config["fine-tune"]["model_name"]
PROCESSED_DATASET_PATH = config["dataset"]["paths"]["processed_dir"]
FINAL_ADAPTERS_OUTPUT_DIR = config["fine-tune"]["adapters_output_dir"]
TRAINER_OUTPUT_DIR = config["fine-tune"]["trainer_output_dir"]

# Make dirs if not exist and if exists then clean up first
make_clean_dir(FINAL_ADAPTERS_OUTPUT_DIR)
make_clean_dir(TRAINER_OUTPUT_DIR)

# --- Step 2: Load Model, Processor, and Quantization Config ---
print(f"--- Loading model: {MODEL_ID} ---")

# Configure 4-bit quantization for memory-efficient training (QLoRA)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the processor (handles both image processing and text tokenization)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Load the base model with the 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Automatically uses the available GPU
)
print("✅ Model and processor loaded successfully.")

# --- Step 3: Configure LoRA (PEFT Adapters) ---
print("\n--- Configuring LoRA for PEFT ---")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)
print("✅ LoRA configured.")

# --- Step 4: Load the Prepared Dataset ---
print(f"\n--- Loading dataset from disk: {PROCESSED_DATASET_PATH} ---")
if not os.path.exists(PROCESSED_DATASET_PATH):
    raise FileNotFoundError(
        f"Processed dataset not found at '{PROCESSED_DATASET_PATH}'. "
        "Please run the data preparation and transformation scripts first."
    )
dataset = load_from_disk(PROCESSED_DATASET_PATH)
dataset = dataset.shuffle(seed=42)  # Shuffle for good measure
print("✅ Dataset loaded and ready for training:")
print(dataset)

# --- Step 5: Configure and Initialize the SFTTrainer ---
print("\n--- Configuring the SFTTrainer ---")
training_args = TrainingArguments(
    output_dir=TRAINER_OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_steps=10,
    optim="paged_adamw_8bit",
    bf16=True,  # Use bfloat16 for modern GPUs
    push_to_hub=False,  # Set to True if you want to upload to Hugging Face Hub
    report_to="tensorboard",
    # For quick testing, uncomment the line below
    # max_steps=100,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    peft_config=lora_config,
    dataset_text_field="messages",
    tokenizer=processor.tokenizer,  # Pass the tokenizer part of the processor
    # Important: The SFTTrainer needs a data collator that can handle images
    # The processor itself can often be used for this if it's a multimodal processor
    # If not, a custom data collator would be needed. For Gemma 3, the processor should work.
    max_seq_length=1024,
)
print("✅ SFTTrainer configured.")

# --- Step 6: Start the Fine-Tuning Process ---
print("\n🚀 --- Starting the fine-tuning process --- 🚀")
trainer.train()
print("\n🏁 --- Fine-tuning complete --- 🏁")

# --- Step 7: Save the Final Model Adapters ---
print(f"\n--- Saving final model adapters to: {FINAL_ADAPTERS_OUTPUT_DIR} ---")
trainer.save_model(FINAL_ADAPTERS_OUTPUT_DIR)
print(f"✅ Fine-tuned model adapters saved successfully.")
