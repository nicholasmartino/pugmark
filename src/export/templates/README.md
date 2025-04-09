# Building Footprint Generator

## Model Description
This model is a pix2pix GAN generator trained to create building footprints from satellite imagery. The model takes 256x256 RGB satellite images as input and produces building footprint masks as output.

## Model Architecture
The generator uses a U-Net architecture with skip connections:
- Encoder: 8 downsampling layers (3→64→128→256→512→512→512→512→512)
- Decoder: 7 upsampling layers with skip connections (512→512→512→256→128→64→3)

## Usage

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.saved_model.load('tf_model')
infer = model.signatures['serving_default']

# Prepare input (normalized to [-1, 1])
image = Image.open('satellite_image.jpg').resize((256, 256))
input_array = np.array(image).astype(np.float32) / 127.5 - 1
input_tensor = tf.convert_to_tensor(input_array[np.newaxis, ...])

# Run inference
output = infer(input_tensor)
output_tensor = list(output.values())[0]

# Convert output to image (from [-1, 1] to [0, 255])
output_array = ((output_tensor[0].numpy() + 1) * 127.5).astype(np.uint8)
output_image = Image.fromarray(output_array)
output_image.save('footprint.png')
```

## Training
This model was trained on satellite imagery from the Metro Vancouver region using a pix2pix GAN architecture.

## License
This model is provided for research and non-commercial use. 