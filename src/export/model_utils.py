import os

import tensorflow as tf

# Fixed path to saved model directory
MODEL_PATH = os.path.join("data", "model")


def load_model():
    """
    Load the model from the fixed path.

    Returns:
        Loaded TensorFlow model or None if loading fails
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Model directory not found at {MODEL_PATH}")
        return None

    # Check if fingerprint.pb exists
    if not os.path.exists(os.path.join(MODEL_PATH, "fingerprint.pb")):
        print(f"fingerprint.pb not found in {MODEL_PATH}")
        return None

    # Load the model
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = tf.saved_model.load(MODEL_PATH)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")

        # Try loading with SavedModel format
        try:
            print("Attempting to load as SavedModel...")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully as SavedModel!")
            return model
        except Exception as e2:
            print(f"Failed to load model as SavedModel: {e2}")
            return None
