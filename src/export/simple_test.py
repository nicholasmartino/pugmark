import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.training.Globals import *
from src.training.Loader import load_image_test

# Path to saved model directory
model_dir = f"{PATH}/footprints/model"


def find_latest_model(use_cache=True, cache_dir="cache/model_cache"):
    """Find the latest saved model with simple caching by model name and timestamp."""
    # Early return if model directory doesn't exist
    if not tf.io.gfile.exists(model_dir):
        print("Model directory not found")
        return None

    # Get all model directories
    subdirs = [
        d
        for d in tf.io.gfile.listdir(model_dir)
        if tf.io.gfile.isdir(os.path.join(model_dir, d))
    ]

    # Early return if no model subdirectories
    if not subdirs:
        print("No model subdirectories found")
        return None

    # Find the latest model directory (model_XXXXX format)
    latest_dir = sorted(
        subdirs,
        key=lambda x: (
            int(x.split("_")[-1].rstrip("/")) if x.startswith("model_") else 0
        ),
    )[-1]

    cloud_model_path = os.path.join(model_dir, latest_dir)

    # If not using cache, return cloud path directly
    if not use_cache:
        return cloud_model_path

    # Get model timestamp for cache key
    try:
        stat_info = tf.io.gfile.stat(cloud_model_path)
        timestamp = (
            stat_info.mtime_nsec
            if hasattr(stat_info, "mtime_nsec")
            else stat_info.mtime
        )
    except Exception as e:
        print(f"Error getting timestamp for {latest_dir}: {e}")
        return None

    # Cache path for this specific model version
    cache_path = os.path.join(cache_dir, f"{latest_dir}_{timestamp}")

    # Use cached version if it exists
    if os.path.exists(cache_path):
        print(f"Using cached model: {cache_path}")
        return cache_path

    # Create cache directory and download model
    print(f"Caching model from cloud to: {cache_path}")
    os.makedirs(cache_path, exist_ok=True)

    # Copy model files to cache
    try:
        files = tf.io.gfile.glob(os.path.join(cloud_model_path, "*"))
        for src_file in files:
            dst_file = os.path.join(cache_path, os.path.basename(src_file))
            tf.io.gfile.copy(src_file, dst_file, overwrite=False)
        return cache_path
    except Exception as e:
        print(f"Error caching model: {e}")
        # Fall back to cloud path if caching fails
        return cloud_model_path


def generate_images(model, test_input, tar, save_path=None):
    """Generate images using the saved model."""
    # For SavedModel objects, we need to handle differently
    try:
        if hasattr(model, "signatures"):
            # Try different signature approaches
            if "serving_default" in model.signatures:
                infer = model.signatures["serving_default"]
                prediction = infer(tf.constant(test_input))
                # Get first output tensor
                prediction = list(prediction.values())[0]
            elif hasattr(model, "serve"):
                prediction = model.serve(test_input)
            else:
                # Last resort - try the call method
                prediction = model(test_input)
        else:
            # Standard model call
            prediction = model(test_input, training=False)
    except Exception as e:
        print(f"Error generating prediction: {e}")
        # If all else fails, attempt a generic predict
        prediction = model.predict(test_input)

    # Convert tensors to numpy arrays for plotting
    test_input_np = (
        test_input[0].numpy() if hasattr(test_input[0], "numpy") else test_input[0]
    )
    tar_np = tar[0].numpy() if hasattr(tar[0], "numpy") else tar[0]

    # Handle prediction based on type
    if isinstance(prediction, dict):
        pred_np = (
            list(prediction.values())[0][0].numpy()
            if hasattr(list(prediction.values())[0][0], "numpy")
            else list(prediction.values())[0][0]
        )
    else:
        pred_np = (
            prediction[0].numpy() if hasattr(prediction[0], "numpy") else prediction[0]
        )

    # Create figure with 3 subplots: input, target, prediction
    plt.figure(figsize=(15, 5))

    display_list = [test_input_np, tar_np, pred_np]
    titles = ["Input Image", "Ground Truth", "Predicted Image"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(titles[i])

        # Normalize images for display
        img = display_list[i]
        img = (img + 1) / 2.0  # Normalize [-1,1] to [0,1]

        # Convert to uint8 for display
        img = (img * 255).astype(np.uint8)

        plt.imshow(img)
        plt.axis("off")

    # Tight layout to avoid overlapping
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, format="png", dpi=200)


def main(num_samples=3, use_cache=True):
    """Test the saved generator model with minimal cloud storage usage."""
    print("Starting model test...")

    # Create output directory
    output_dir = "cache/test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Find and load the latest model
    model_path = find_latest_model(use_cache=use_cache)
    if not model_path:
        print("No model found to test")
        return

    # Load the model
    print(f"Loading model from: {model_path}")
    try:
        generator = tf.saved_model.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Model loaded successfully!")

    # Load test dataset
    test_dataset = tf.data.Dataset.list_files(f"{PATH}/footprints/test/*.png")
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # Generate test images
    print(f"Generating {num_samples} test images...")

    for i, (inp, tar) in enumerate(test_dataset.take(num_samples)):
        print(f"Image {i+1}/{num_samples}")
        output_path = os.path.join(output_dir, f"test_output_{i+1}.png")
        generate_images(generator, inp, tar, save_path=output_path)

    print(f"Done! Images saved to {output_dir}/")


if __name__ == "__main__":
    main()
