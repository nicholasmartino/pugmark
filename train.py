import os

import tensorflow as tf
from datasets import load_dataset
from transformers import TrainingArguments

from pix2pix import Discriminator, Generator, Pix2PixTrainer, prepare_dataset


def main():
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

    # Prepare datasets
    dataset = load_dataset("nicholasmartino/building-footprints")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Initialize models
    generator = Generator()
    discriminator = Discriminator()

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
