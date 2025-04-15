import os

import matplotlib.pyplot as plt
import tensorflow as tf

from src.training.Generator import generate_images
from src.training.Globals import *
from src.training.Loader import load_image_test

# Path to saved model directory
model_dir = f"{PATH}/footprints/model"


def find_latest_model():
    """Find the latest saved model in the model directory."""
    if not tf.io.gfile.exists(model_dir):
        return None

    try:
        # Get all model directories
        subdirs = [
            d
            for d in tf.io.gfile.listdir(model_dir)
            if tf.io.gfile.isdir(os.path.join(model_dir, d))
        ]

        if not subdirs:
            return None

        # Find the latest model directory (model_XXXXX format)
        latest_dir = sorted(
            subdirs,
            key=lambda x: int(x.split("_")[-1]) if x.startswith("model_") else 0,
        )[-1]

        return os.path.join(model_dir, latest_dir)
    except Exception as e:
        print(f"Error finding latest model: {e}")
        return None


def main():
    """Test the saved generator model."""
    print("Starting model test...")

    # Create output directory
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Find and load the latest model
    latest_model_path = find_latest_model()
    if not latest_model_path:
        print(f"Error: No model found in {model_dir}")
        return

    print(f"Loading model from: {latest_model_path}")
    try:
        generator = tf.saved_model.load(latest_model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load test dataset
    test_dataset = tf.data.Dataset.list_files(f"{PATH}/footprints/test/*.png")
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # Generate test images
    print("Generating images from test dataset...")
    plt.switch_backend("agg")

    for i, (inp, tar) in enumerate(test_dataset.take(5)):
        print(f"Generating image {i+1}/5")
        generate_images(generator, inp, tar)
        plt.savefig(os.path.join(output_dir, f"test_output_{i+1}.png"))
        plt.close()

    print(f"Testing complete! Generated images saved to {output_dir}/")


if __name__ == "__main__":
    main()
