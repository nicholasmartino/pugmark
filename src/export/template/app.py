import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the model
try:
    model = tf.saved_model.load("tf_model")
    if "serving_default" in model.signatures:
        infer = model.signatures["serving_default"]
    else:
        # Use first available signature if serving_default is not available
        infer = list(model.signatures.values())[0]
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise


def process_image(input_image):
    """Process an input satellite image to generate building footprints"""
    if input_image is None:
        return None

    # Resize to 256x256
    img = Image.fromarray(input_image).resize((256, 256))

    # Normalize to [-1, 1]
    input_array = np.array(img).astype(np.float32) / 127.5 - 1
    input_tensor = tf.convert_to_tensor(input_array[np.newaxis, ...])

    # Run inference
    try:
        output = infer(input_tensor)
        output_tensor = list(output.values())[0]

        # Convert output to image (from [-1, 1] to [0, 255])
        output_array = ((output_tensor[0].numpy() + 1) * 127.5).astype(np.uint8)

        return output_array
    except Exception as e:
        print(f"Error during inference: {e}")
        return None


# Create the Gradio interface
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("# Metro Vancouver Building Footprint Generator")
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
        examples=[
            "examples/example1.jpg",
            "examples/example2.jpg",
        ],
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
