import os

import torch
from accelerate import Accelerator
from datasets import load_dataset
from transformers import TrainingArguments
import shutil
import random

from pix2pix_hf import Discriminator, Generator, Pix2PixDataset, Pix2PixTrainer


def main():
    # Initialize accelerator
    accelerator = Accelerator()

    # Configuration
    # PATH should point to the dataset on Hugging Face or mounted storage
    PATH = os.getenv(
        "DATASET_PATH", "path/to/your/dataset/"
    )  # Make configurable via env var
    EPOCHS = int(os.getenv("EPOCHS", "10"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir="/tmp/pix2pix_results",  # Use tmp directory on HF infrastructure
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_dir="/tmp/logs",
        logging_steps=100,
        save_steps=1000,
        evaluation_strategy="epoch",
        report_to="wandb",
        # Add these for better performance on HF infrastructure
        fp16=True,  # Enable mixed precision training
        gradient_checkpointing=True,  # Reduce memory usage
        gradient_accumulation_steps=4,  # Helps with memory constraints
    )

    # Load and prepare datasets
    dataset = load_dataset("nicholasmartino/building-footprints")

    # Create Pix2PixDataset instances
    train_dataset = Pix2PixDataset(dataset["train"]["image"], split="train")
    test_dataset = Pix2PixDataset(dataset["test"]["image"], split="test")

    # Initialize models with PyTorch
    generator = Generator()
    discriminator = Discriminator()

    # Move models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # Prepare everything with accelerator
    generator, discriminator, train_dataset, test_dataset = accelerator.prepare(
        generator, discriminator, train_dataset, test_dataset
    )

    # Create trainer
    trainer = Pix2PixTrainer(
        generator=generator,
        discriminator=discriminator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train the model
    trainer.train()

    # Save models (optional)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
