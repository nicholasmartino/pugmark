import os

import matplotlib.pyplot as plt
import tensorflow as tf

# Import directly from the trainer module to ensure identical model instances
from src.training.Globals import *
from src.training.Loader import load_image_test
from src.training.Trainer import (
    discriminator,
    discriminator_optimizer,
    generate_images,
    generator,
    generator_optimizer,
)

# Path to checkpoint directory
checkpoint_dir = f"{PATH}/footprints/checkpoint"


def main():
    """Test the model with the latest checkpoint"""
    print("Starting model test...")

    # Check if checkpoint directory exists
    if not tf.io.gfile.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory {checkpoint_dir} does not exist.")
        return

    # Use the same model instances from the trainer module
    print("Using models from trainer module")

    # Create epoch and step counters for checkpoint compatibility
    epoch_counter = tf.Variable(
        0, trainable=False, dtype=tf.int64, name="epoch_counter"
    )
    step_counter = tf.Variable(0, trainable=False, dtype=tf.int64, name="step_counter")

    # Set up checkpoint using the same models as training
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )

    # Try to restore the checkpoint
    try:
        # Check if best_model exists and use that first
        best_model_path = os.path.join(checkpoint_dir, "best_model")
        if tf.io.gfile.exists(best_model_path + ".index"):
            print("Best model checkpoint found. Using that.")
            checkpoint.restore(best_model_path)
        else:
            # Otherwise use latest checkpoint
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                print(f"Restoring latest checkpoint: {latest_checkpoint}")
                checkpoint.restore(latest_checkpoint)
            else:
                print("No valid checkpoint found.")
                return

        print("Model restored successfully")
    except Exception as e:
        print(f"Error restoring checkpoint: {e}")
        return

    # Load test dataset
    test_dataset = tf.data.Dataset.list_files(f"{PATH}/footprints/test/*.png")
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # Create output directory for test images
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)

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
