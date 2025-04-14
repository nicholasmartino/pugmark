import os

import matplotlib.pyplot as plt
import tensorflow as tf

from src.training.Discriminator import Discriminator
from src.training.Generator import Generator, generate_images
from src.training.Globals import *
from src.training.Loader import load_image_test

# Path to checkpoint directory
checkpoint_dir = f"{PATH}/footprints/checkpoint"


def main():
    """Test the model with the latest checkpoint"""
    print("Starting model test...")

    # Check if checkpoint directory exists
    if not tf.io.gfile.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory {checkpoint_dir} does not exist.")
        return

    # Create models with same architecture as during training
    generator = Generator()
    discriminator = Discriminator()

    print("Models initialized")

    # Create optimizers and checkpoint variables
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    epoch_counter = tf.Variable(
        0, trainable=False, dtype=tf.int64, name="epoch_counter"
    )
    step_counter = tf.Variable(0, trainable=False, dtype=tf.int64, name="step_counter")

    # Set up checkpoint
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
        epoch_counter=epoch_counter,
        step_counter=step_counter,
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

        print(
            f"Model restored to epoch {epoch_counter.numpy()}, step {step_counter.numpy()}"
        )
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
