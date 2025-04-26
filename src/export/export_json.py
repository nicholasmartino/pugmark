import os

import tensorflow as tf
import tensorflowjs as tfjs
from Generator import Generator
from Globals import PATH


def export_model(checkpoint_dir):
    """Export the trained model to TensorFlow.js format."""
    # Initialize the generator
    generator = Generator()

    # Create checkpoint
    checkpoint = tf.train.Checkpoint(generator=generator)

    # Restore the latest checkpoint
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f"Restoring from checkpoint: {latest_checkpoint}")
        checkpoint.restore(latest_checkpoint)
    else:
        raise ValueError("No checkpoint found to export")

    # Save the model in HDF5 format first
    h5_path = os.path.join(os.getcwd(), "model.h5")
    generator.save(h5_path)

    # Convert the HDF5 model to TensorFlow.js format
    tfjs.converters.convert_keras_model_to_tfjs_layers_model(
        h5_path, os.getcwd(), input_format="keras"
    )

    # Clean up the temporary HDF5 file
    os.remove(h5_path)

    print("Model exported successfully!")


if __name__ == "__main__":
    checkpoint_dir = f"{PATH}/footprints/checkpoint"
    export_model(checkpoint_dir)
