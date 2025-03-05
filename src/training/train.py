import logging
import os

import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Log TensorFlow version and available devices
logger.info(f"TensorFlow version: {tf.__version__}")
physical_devices = tf.config.list_physical_devices()
logger.info(f"Available devices: {physical_devices}")

# Check for GPUs
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        logger.info(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logger.error(f"GPU memory growth configuration error: {e}")
else:
    logger.warning("No GPU found. Training will be slow on CPU.")

# Set mixed precision policy for faster training on GPUs
if gpus:
    logger.info("Using mixed precision training")
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)

# Check if running in Vertex AI
is_vertex_ai = os.environ.get("AIP_MODE") == "training"
if is_vertex_ai:
    logger.info("Running in Vertex AI training environment")
    # Set any Vertex AI specific configurations here
else:
    logger.info("Running in standard environment")


if __name__ == "__main__":
    try:
        from Trainer import train

        # Run the training function
        logger.info("Starting training process")
        train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
