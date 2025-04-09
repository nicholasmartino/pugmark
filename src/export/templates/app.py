import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the model
model = tf.saved_model.load('tf_model')
infer = model.signatures['serving_default']

def process_image(input_image):
    # Resize to 256x256
    img = Image.fromarray(input_image).resize((256, 256))
    
    # Normalize to [-1, 1]
    input_array = np.array(img).astype(np.float32) / 127.5 - 1
    input_tensor = tf.convert_to_tensor(input_array[np.newaxis, ...])
    
    # Run inference
    output = infer(input_tensor)
    output_tensor = list(output.values())[0]
    
    # Convert output to image (from [-1, 1] to [0, 255])
    output_array = ((output_tensor[0].numpy() + 1) * 127.5).astype(np.uint8)
    
    return output_array

iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="Building Footprint Generator",
    description="Upload a satellite image to generate building footprints"
)

iface.launch() 