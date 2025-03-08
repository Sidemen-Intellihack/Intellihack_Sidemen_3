#!/usr/bin/env python
"""
Script to fine-tune Qwen2.5-3B-Instruct using QLoRA with 4-bit quantization.
Requirements:
    - transformers
    - datasets
    - peft
    - bitsandbytes
Install them with:
    pip install transformers datasets peft bitsandbytes
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def main():
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -----------------------------
    # Model and Quantization Setup
    # -----------------------------
    model_name = "Qwen2.5-3B-Instruct"  # Update this to the correct HF model name if needed

    # Define the 4-bit quantization configuration using BitsAndBytes
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",         # Use "nf4" for improved accuracy.
        bnb_4bit_use_double_quant=True,      # Enable double quantization.
        bnb_4bit_compute_dtype=torch.float16  # Use float16 for compute.
    )

    # Load the tokenizer. Ensure the tokenizer has a pad token.
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the model with 4-bit quantization.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )

    # Prepare the model for k-bit training (enabling QLoRA).
    model = prepare_model_for_kbit_training(model)

    # Define LoRA configuration.
    # Adjust target_modules based on the model architecture.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Update these if the model architecture is different.
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Wrap the model with PEFT using the LoRA configuration.
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Print the trainable parameters for verification.

    # -----------------------------
    # Dataset Preparation
    # -----------------------------
    # Specify the path to your JSONL dataset file.
    # The JSONL file should have one JSON object per line with keys: "instruction", "input", and "output".
    dataset_path = "path/to/your/dataset.jsonl"  # <-- Update this path

    # Load the dataset using the Hugging Face Datasets library.
    dataset = load_dataset("json", data_files={"train": dataset_path})["train"]

    # Define a tokenization function. Here we format each example by combining:
    # 1. The instruction
    # 2. The input (if available)
    # 3. The output (appended after a prompt separator)
    def tokenize_function(examples):
        texts = []
        for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            if inp and inp.strip():
                # Format with both instruction and input.
                prompt = f"Instruction: {instr}\nInput: {inp}\nResponse:"
            else:
                # Format with only the instruction.
                prompt = f"Instruction: {instr}\nResponse:"
            # Append the expected output as part of the complete text.
            full_text = prompt + " " + out
            texts.append(full_text)
        # Tokenize the concatenated texts.
        tokenized_output = tokenizer(
            texts, truncation=True, max_length=512, padding="max_length"
        )
        # For causal LM training, the labels are the same as the input IDs.
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()
        return tokenized_output

    # Apply the tokenization to the dataset.
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,  # Remove raw columns to save space.
    )

    # -----------------------------
    # Training Configuration
    # -----------------------------
    training_args = TrainingArguments(
        output_dir="./qwen2.5-3b-qlora-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,                    # Set number of epochs.
        per_device_train_batch_size=8,         # Adjust the batch size based on your GPU memory.
        learning_rate=2e-4,                    # Learning rate.
        fp16=True,                             # Use fp16 mixed precision.
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="no",              # No evaluation during training.
        save_total_limit=2,
        dataloader_num_workers=4,
        report_to="none",                      # Disable logging to third-party tools.
    )

    # Data collator to dynamically pad the inputs during training.
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # -----------------------------
    # Fine-Tuning with Trainer API
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Start the training process.
    trainer.train()

    # Save the fine-tuned model and tokenizer.
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"Model and tokenizer saved to {training_args.output_dir}")

    # -----------------------------
    # Inference Function
    # -----------------------------
    def run_inference(prompt_text, max_new_tokens=100):
        """Generates and prints a response for a given prompt."""
        # Tokenize the input prompt.
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        # Generate output text.
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
        # Decode the generated tokens.
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    # Test the inference function with a sample prompt.
    test_prompt = "Instruction: Summarize the impact of 3PL on supply chain efficiency.\nResponse:"
    print("\nInference Test:")
    print(run_inference(test_prompt))

if _name_ == "_main_":
    main()