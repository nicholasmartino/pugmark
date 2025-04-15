import os

import matplotlib.pyplot as plt
import tensorflow as tf

from src.training.Generator import generate_images

# Import only what we need for testing
from src.training.Globals import *
from src.training.Loader import load_image_test

# Path to saved model directory
model_dir = f"{PATH}/footprints/model"
# Fallback to checkpoint directory
checkpoint_dir = f"{PATH}/footprints/checkpoint"


def main():
    """Test the model with saved model or fallback to checkpoint"""
    print("Starting model test...")

    # Create output directory for test images
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load test dataset
    test_dataset = tf.data.Dataset.list_files(f"{PATH}/footprints/test/*.png")
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # First, try to find the latest saved model
    generator = None
    if tf.io.gfile.exists(model_dir):
        # List all subdirectories in the model directory
        try:
            subdirs = [
                d
                for d in tf.io.gfile.listdir(model_dir)
                if tf.io.gfile.isdir(os.path.join(model_dir, d))
            ]

            if subdirs:
                # Sort directories to find the latest model (model_XXXXX format)
                latest_model_dir = sorted(
                    subdirs,
                    key=lambda x: (
                        int(x.split("_")[-1]) if x.startswith("model_") else 0
                    ),
                )[-1]
                latest_model_path = os.path.join(model_dir, latest_model_dir)

                print(f"Loading saved model from: {latest_model_path}")
                generator = tf.saved_model.load(latest_model_path)
                print("Generator model loaded successfully!")
        except Exception as e:
            print(f"Error finding/loading saved model: {e}")
            generator = None

    # If saved model couldn't be loaded, try checkpoint as fallback
    if generator is None:
        print(
            "No saved model found or loading failed. Trying checkpoint as fallback..."
        )

        # Import what we need for checkpoint loading
        from src.training.Discriminator import Discriminator
        from src.training.Generator import Generator

        # Create model instances
        generator = Generator()
        discriminator = Discriminator()

        # Create optimizers with the same parameters as in training
        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        # Create a checkpoint for loading
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            discriminator=discriminator,
        )

        # Initialize the model with a forward pass
        for sample_input, _ in test_dataset.take(1):
            print(f"Model initialization input shape: {sample_input.shape}")
            # Force model initialization with forward pass
            _ = generator(sample_input, training=False)
            break

        try:
            # Look for the latest checkpoint
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                print(f"Restoring from checkpoint: {latest_checkpoint}")
                checkpoint.restore(latest_checkpoint).expect_partial()
                print("Checkpoint restored successfully")
            else:
                print("No checkpoint found.")
                return
        except Exception as e:
            print(f"Error restoring checkpoint: {e}")
            return

    # Generate images using test samples
    print("Generating images from test dataset...")
    for i, (inp, tar) in enumerate(test_dataset.take(5)):
        print(f"Generating image {i+1}/5")
        # Use matplotlib backend that doesn't require display
        plt.switch_backend("agg")
        generate_images(generator, inp, tar)
        plt.savefig(os.path.join(output_dir, f"test_output_{i+1}.png"))
        plt.close()

    print(f"Testing complete! Generated images saved to {output_dir}/")


if __name__ == "__main__":
    main()
