import os

import tensorflow as tf

from src.training.Globals import PATH

# Path to saved model directory
MODEL_DIR = f"{PATH}/footprints/model"
DEFAULT_CACHE_DIR = "cache/model_cache"


def find_latest_model(use_cache=True, cache_dir=DEFAULT_CACHE_DIR):
    """
    Find the latest saved model with simple caching by model name and timestamp.
    Used by both testing and export functionality.

    Args:
        use_cache: Whether to cache the model locally
        cache_dir: Directory to store cached models

    Returns:
        Path to the model (either cached or original cloud path)
    """
    # Early return if model directory doesn't exist
    if not tf.io.gfile.exists(MODEL_DIR):
        print("Model directory not found")
        return None

    # Get all model directories
    subdirs = [
        d
        for d in tf.io.gfile.listdir(MODEL_DIR)
        if tf.io.gfile.isdir(os.path.join(MODEL_DIR, d))
    ]

    # Early return if no model subdirectories
    if not subdirs:
        print("No model subdirectories found")
        return None

    # Find the latest model directory (model_XXXXX format)
    latest_dir = sorted(
        subdirs,
        key=lambda x: (
            int(x.split("_")[-1].rstrip("/")) if x.startswith("model_") else 0
        ),
    )[-1]

    cloud_model_path = os.path.join(MODEL_DIR, latest_dir)

    # If not using cache, return cloud path directly
    if not use_cache:
        return cloud_model_path

    # Get model timestamp for cache key
    try:
        stat_info = tf.io.gfile.stat(cloud_model_path)
        timestamp = (
            stat_info.mtime_nsec
            if hasattr(stat_info, "mtime_nsec")
            else stat_info.mtime
        )
    except Exception as e:
        print(f"Error getting timestamp for {latest_dir}: {e}")
        return None

    # Cache path for this specific model version
    cache_path = os.path.join(cache_dir, f"{latest_dir}_{timestamp}")

    # Use cached version if it exists
    if os.path.exists(cache_path):
        print(f"Using cached model: {cache_path}")
        return cache_path

    # Create cache directory and download model
    print(f"Caching model from cloud to: {cache_path}")
    os.makedirs(cache_path, exist_ok=True)

    # Copy model files to cache
    try:
        files = tf.io.gfile.glob(os.path.join(cloud_model_path, "*"))
        for src_file in files:
            dst_file = os.path.join(cache_path, os.path.basename(src_file))
            tf.io.gfile.copy(src_file, dst_file, overwrite=False)
        return cache_path
    except Exception as e:
        print(f"Error caching model: {e}")
        # Fall back to cloud path if caching fails
        return cloud_model_path


def load_model(model_path=None, use_cache=True, cache_dir=DEFAULT_CACHE_DIR):
    """
    Load the latest model, with optional caching.

    Args:
        model_path: Optional specific model path to load
        use_cache: Whether to use cached model
        cache_dir: Directory for cached models

    Returns:
        Loaded TensorFlow model or None if loading fails
    """
    # Find the model if path not provided
    if model_path is None:
        model_path = find_latest_model(use_cache=use_cache, cache_dir=cache_dir)

    if not model_path:
        print("No model found to load")
        return None

    # Load the model
    print(f"Loading model from: {model_path}")
    try:
        model = tf.saved_model.load(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
