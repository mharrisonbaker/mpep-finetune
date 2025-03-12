import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import (
    TrainingArguments, 
    TextStreamer,
    LlamaTokenizer,
    LlamaForCausalLM
)
from peft import LoraConfig, get_peft_model
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model_and_tokenizer(model_path: str, max_seq_length: int = 2048):
    """Load the model and tokenizer with proper quantization"""
    logger.info("Loading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(
        model_path,
        model_max_length=max_seq_length,
        padding_side="right",
        use_fast=False,
    )
    
    # Set special tokens
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Loading model...")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_8bit=True  # Using 8-bit quantization for highest possible precision
    )
    
    # Configure LoRA
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def format_instruction(instruction: str, response: str) -> str:
    """Format the instruction and response into a chat template"""
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}</s>"

def prepare_dataset(dataset_path: str, tokenizer):
    """Load and prepare the dataset"""
    logger.info("Loading dataset...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    def preprocess_function(examples):
        texts = [
            format_instruction(instr, resp)
            for instr, resp in zip(examples["instruction"], examples["response"])
        ]
        return {"text": texts}
    
    logger.info("Preprocessing dataset...")
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return processed_dataset

def train_model(model, tokenizer, dataset, output_dir: str, max_seq_length: int = 2048):
    """Configure and run the training"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        bf16=True,  # Use bfloat16 if available
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True,
        args=training_args
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    return trainer

def save_model(trainer, output_dir: str):
    """Save the trained model"""
    logger.info(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    
def main():
    # Configuration
    model_path = "/home/ubuntu/llama-8b"
    dataset_path = "/home/ubuntu/mpep_data.jsonl"
    output_dir = "/home/ubuntu/fine_tuned_output"
    max_seq_length = 2048
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Setup
        model, tokenizer = setup_model_and_tokenizer(model_path, max_seq_length)
        
        # Prepare dataset
        dataset = prepare_dataset(dataset_path, tokenizer)
        
        # Train
        trainer = train_model(model, tokenizer, dataset, output_dir, max_seq_length)
        
        # Save
        save_model(trainer, output_dir)
        
        logger.info("Training complete!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()