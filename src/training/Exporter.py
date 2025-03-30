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

    # Export to TensorFlow.js format
    tfjs.converters.save_keras_model(generator, os.getcwd())
    print("Model exported successfully!")


if __name__ == "__main__":
    checkpoint_dir = f"{PATH}/footprints/checkpoint"
    export_model(checkpoint_dir)
