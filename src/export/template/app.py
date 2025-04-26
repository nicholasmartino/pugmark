import os
import traceback

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# Path to the model directory
MODEL_DIR = "tf_model/model_40000/_0"


def load_model():
    try:
        if not os.path.exists(MODEL_DIR):
            print(f"Model directory '{MODEL_DIR}' does not exist")
            return None

        files = os.listdir(MODEL_DIR)
        print(f"Files in model directory: {files}")

        model = tf.saved_model.load(MODEL_DIR)
        print("Model loaded successfully!")

        # Print model info
        print("Model structure:", model)
        if hasattr(model, "signatures"):
            print("Available signatures:", list(model.signatures.keys()))

            # Get the default signature
            serving_default = model.signatures["serving_default"]

            # Print input specs
            print("\nModel Input Specifications:")
            for input_name, input_tensor in serving_default.structured_input_signature[
                1
            ].items():
                print(
                    f"Input '{input_name}': shape={input_tensor.shape}, dtype={input_tensor.dtype}"
                )

            # Print output specs
            print("\nModel Output Specifications:")
            for (
                output_name,
                output_tensor,
            ) in serving_default.structured_outputs.items():
                print(
                    f"Output '{output_name}': shape={output_tensor.shape}, dtype={output_tensor.dtype}"
                )

        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# Load the model
model = load_model()


def process_image(input_image):
    try:
        print(f"Received input of type: {type(input_image)}")
        if input_image is None:
            print("No input image provided.")
            return None
        # Validate input is a numpy array
        if not isinstance(input_image, np.ndarray):
            raise ValueError(f"Input is not a numpy array. Got: {type(input_image)}")
        print(f"Input image shape: {input_image.shape}, dtype: {input_image.dtype}")
        # Ensure 3 channels
        if input_image.ndim == 2:
            input_image = np.stack([input_image] * 3, axis=-1)
        elif input_image.shape[-1] == 4:
            input_image = input_image[..., :3]
        # Convert to PIL Image
        img = Image.fromarray(input_image.astype(np.uint8))
        # Resize to model input size (256x256)
        model_input_size = (256, 256)
        img = img.resize(model_input_size, Image.Resampling.LANCZOS)
        print(f"Resized to {model_input_size} for model input")
        # Convert to numpy array and normalize to [-1, 1]
        img_array = np.array(img).astype(np.float32)
        img_array = (img_array / 127.5) - 1.0
        print(
            f"Normalized array shape: {img_array.shape}, Range: [{img_array.min():.2f}, {img_array.max():.2f}]"
        )
        # Add batch dimension
        img_array = np.expand_dims(img_array, 0)
        print(f"Input tensor shape: {img_array.shape}")
        # Run inference
        if model is None:
            print("Model not loaded!")
            return None
        print("Running model inference...")
        # Use the default signature if available
        if hasattr(model, "signatures") and "serving_default" in model.signatures:
            infer = model.signatures["serving_default"]
            predictions = infer(tf.constant(img_array))
            # Get first output tensor
            predictions = list(predictions.values())[0]
        else:
            predictions = model(img_array)
        print(f"Raw prediction shape: {predictions.shape}")
        # Denormalize output to [0, 255]
        output = ((predictions[0] + 1.0) * 127.5).numpy().clip(0, 255).astype(np.uint8)
        print(
            f"Denormalized output shape: {output.shape}, Range: [{output.min()}, {output.max()}]"
        )
        # Resize to final output size (128x128 -> 256x256 for display)
        output_size = (256, 256)
        output_img = Image.fromarray(output).resize(
            output_size, Image.Resampling.LANCZOS
        )
        print(f"Final output size: {output_img.size}")
        return np.array(output_img)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        traceback.print_exc()
        return None


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
    gr.Markdown("# Building Footprint Generator")

    if model is None:
        gr.Markdown(
            "⚠️ **Model not loaded!** The app will run but cannot generate predictions."
        )

    gr.Markdown(
        """
        Upload a satellite image to generate building footprints. 
        
        **Instructions:**
        1. Upload a satellite image (ideally 256x256 pixels)
        2. Click 'Generate Footprints'
        3. The model will generate building outlines
        
        The model works best with RGB satellite images of urban areas.
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input Satellite Image", type="numpy", height=256, width=256
            )
            submit_btn = gr.Button("Generate Footprints")
        with gr.Column():
            output_image = gr.Image(
                label="Building Footprints", type="numpy", height=256, width=256
            )

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
