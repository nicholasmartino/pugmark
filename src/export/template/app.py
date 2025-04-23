import os

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# Path to the model directory
MODEL_DIR = "tf_model"


# Load the model with better error handling
def load_model():
    try:
        # Check if model directory exists
        if not os.path.exists(MODEL_DIR):
            print(f"Model directory '{MODEL_DIR}' does not exist")
            return None

        # List model directory contents
        files = os.listdir(MODEL_DIR)
        print(f"Files in model directory: {files}")

        # Check for saved_model.pb
        if "saved_model.pb" not in files:
            print(f"saved_model.pb not found in {MODEL_DIR}")
            return None

        # Load the model
        model = tf.saved_model.load(MODEL_DIR)
        print("Model loaded successfully!")

        # Get appropriate signature
        if "serving_default" in model.signatures:
            infer = model.signatures["serving_default"]
            print("Using 'serving_default' signature")
        else:
            # Use first available signature
            sig_keys = list(model.signatures.keys())
            if sig_keys:
                infer = model.signatures[sig_keys[0]]
                print(f"Using '{sig_keys[0]}' signature")
            else:
                print("No signatures found in model")
                return None

        return infer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# Load the model
inference_fn = load_model()


def process_image(input_image):
    """Process an input satellite image to generate building footprints"""
    # Check if model loaded successfully
    if inference_fn is None:
        print("Model not available for inference")
        return np.zeros((256, 256, 3), dtype=np.uint8)

    if input_image is None:
        return np.zeros((256, 256, 3), dtype=np.uint8)

    try:
        # Resize to 256x256
        img = Image.fromarray(input_image).resize((256, 256))

        # Normalize to [-1, 1]
        input_array = np.array(img).astype(np.float32) / 127.5 - 1
        input_tensor = tf.convert_to_tensor(input_array[np.newaxis, ...])

        # Run inference
        output = inference_fn(input_tensor)
        output_tensor = list(output.values())[0]

        # Convert output to image (from [-1, 1] to [0, 255])
        output_array = ((output_tensor[0].numpy() + 1) * 127.5).astype(np.uint8)
        return output_array
    except Exception as e:
        print(f"Error during inference: {e}")
        # Return blank image on error
        return np.zeros((256, 256, 3), dtype=np.uint8)


# Create a placeholder image for examples
placeholder_path = os.path.join(os.path.dirname(__file__), "examples")
os.makedirs(placeholder_path, exist_ok=True)

# If example images don't exist, create empty placeholders
example_paths = []
for i in range(1, 3):
    example_path = os.path.join(placeholder_path, f"example{i}.jpg")
    example_paths.append(example_path)
    if not os.path.exists(example_path):
        # Create a blank example image
        blank = Image.new("RGB", (256, 256), color=(240, 240, 240))
        blank.save(example_path)

# Create the Gradio interface
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("# Metro Vancouver Building Footprint Generator")

    if inference_fn is None:
        gr.Markdown(
            "⚠️ **Model not loaded!** The app will run but cannot generate predictions."
        )

    gr.Markdown(
        "Upload a satellite image to generate building footprints. The model works best with RGB images of urban areas."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Satellite Image", type="numpy")
            submit_btn = gr.Button("Generate Footprints")
        with gr.Column():
            output_image = gr.Image(label="Building Footprints", type="numpy")

    # Examples
    gr.Markdown("## Examples")
    examples = gr.Examples(
        examples=example_paths,
        inputs=input_image,
        outputs=output_image,
        fn=process_image,
        cache_examples=True,
    )

    submit_btn.click(fn=process_image, inputs=input_image, outputs=output_image)

    gr.Markdown("## How It Works")
    gr.Markdown(
        """
    This application uses a pix2pix GAN model trained on Metro Vancouver satellite imagery to identify building footprints.
    
    The model was trained on paired images of satellite photos and corresponding building footprints.
    It's designed to work with 256x256 RGB images and produces building outline masks.
    """
    )

# Launch the app
demo.launch()
