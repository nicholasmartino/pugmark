import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

# Set the path to the exported model
model_dir = os.path.join("data", "hf_export", "tf_model")
print(f"Loading model from {model_dir}")

# Load the model
model = tf.saved_model.load(model_dir)
infer = model.signatures["serving_default"]


def process_image(input_path, output_path):
    """
    Process an image file and save the output

    Args:
        input_path: Path to input image
        output_path: Path to save output image
    """
    # Load and resize image
    img = Image.open(input_path).resize((256, 256))
    input_array = np.array(img).astype(np.float32)

    # Normalize to [-1, 1]
    input_array = input_array / 127.5 - 1
    input_tensor = tf.convert_to_tensor(input_array[np.newaxis, ...])

    # Run inference
    print("Running inference...")
    output = infer(input_tensor)
    output_tensor = list(output.values())[0]

    # Convert output to image (from [-1, 1] to [0, 255])
    output_array = ((output_tensor[0].numpy() + 1) * 127.5).astype(np.uint8)

    # Save output
    output_img = Image.fromarray(output_array)
    output_img.save(output_path)
    print(f"Saved output to {output_path}")

    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Generated Footprints")
    plt.imshow(output_array)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(output_path), "comparison.png"))
    print(
        f"Saved comparison to {os.path.join(os.path.dirname(output_path), 'comparison.png')}"
    )
    plt.show()


if __name__ == "__main__":
    # Check if sample input exists, otherwise create dummy input
    test_dir = os.path.join("data", "test")
    os.makedirs(test_dir, exist_ok=True)

    # Use dummy input if no sample is provided
    input_path = os.path.join(test_dir, "sample_input.png")
    output_path = os.path.join(test_dir, "sample_output.png")

    if not os.path.exists(input_path):
        print("No sample input found, creating a dummy image...")
        # Create dummy image with white background and greenish-blue square
        dummy = np.ones((256, 256, 3), dtype=np.uint8) * 255  # White background
        # Add greenish-blue square (RGB: 64, 224, 208 - turquoise)
        dummy[64:192, 64:192, 0] = 64  # Red channel
        dummy[64:192, 64:192, 1] = 224  # Green channel
        dummy[64:192, 64:192, 2] = 208  # Blue channel
        dummy_img = Image.fromarray(dummy)
        dummy_img.save(input_path)
        print(f"Created dummy input at {input_path}")

    # Process the image
    process_image(input_path, output_path)
