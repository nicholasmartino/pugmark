import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.export.model_utils import load_model
from src.training.Globals import *
from src.training.Loader import load_image_test


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

    # Load the model using the centralized utility
    generator = load_model(use_cache=use_cache)
    if generator is None:
        return

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
