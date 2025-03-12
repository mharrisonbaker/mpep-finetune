import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    logging
)
from peft import LoraConfig, get_peft_model
import torch

# Set up logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")

# Configuration
MODEL_PATH = "/home/ubuntu/llama-3.1-8b"
DATA_PATH = "/home/ubuntu/mpep_data.jsonl"
OUTPUT_DIR = "/home/ubuntu/fine_tuned_llama"
BATCH_SIZE = 1
MAX_LENGTH = 1024
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
GRAD_ACCUM_STEPS = 16

# Load dataset
logger.info("Loading dataset...")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# Tokenizer and model
logger.info("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Apply LoRA for fine-tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Preprocess dataset
def preprocess_function(examples):
    combined_texts = [
        f"### Instruction:\n{instruction}\n\n### Response:\n{response}\n"
        for instruction, response in zip(examples["instruction"], examples["response"])
    ]
    tokenized = tokenizer(
        combined_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    return tokenized

logger.info("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none"
)

# Trainer
logger.info("Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

# Save the fine-tuned model
logger.info("Saving fine-tuned model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
logger.info(f"Fine-tuning complete! Model saved to {OUTPUT_DIR}")
