import os

import tensorflow as tf
import tensorflowjs as tfjs
from Globals import CHANNELS, PATH


def create_generator_model():
    """Create a generator model matching the checkpoint architecture."""
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    # Downsample layers
    down_stack = [
        # Layer 1: 3 -> 64 channels
        tf.keras.layers.Conv2D(
            64,
            4,
            strides=2,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
            use_bias=False,
        ),
        # Layer 2: 64 -> 128 channels
        tf.keras.Sequential(
            [
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Conv2D(
                    128,
                    4,
                    strides=2,
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
            ]
        ),
        # Layer 3: 128 -> 256 channels
        tf.keras.Sequential(
            [
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Conv2D(
                    256,
                    4,
                    strides=2,
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
            ]
        ),
        # Layer 4: 256 -> 512 channels
        tf.keras.Sequential(
            [
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Conv2D(
                    512,
                    4,
                    strides=2,
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
            ]
        ),
        # Layers 5-8: 512 -> 512 channels
        *[
            tf.keras.Sequential(
                [
                    tf.keras.layers.LeakyReLU(0.2),
                    tf.keras.layers.Conv2D(
                        512,
                        4,
                        strides=2,
                        padding="same",
                        kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
                        use_bias=False,
                    ),
                    (
                        tf.keras.layers.BatchNormalization()
                        if i < 3
                        else tf.keras.layers.Identity()
                    ),
                ]
            )
            for i in range(4)
        ],
    ]

    # Upsample layers
    up_stack = [
        # First 3 upsamples: 512 -> 512 with dropout
        *[
            tf.keras.Sequential(
                [
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2DTranspose(
                        512,
                        4,
                        strides=2,
                        padding="same",
                        kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
                        use_bias=False,
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.5),
                ]
            )
            for _ in range(3)
        ],
        # 4th upsample: 512 -> 256
        tf.keras.Sequential(
            [
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2DTranspose(
                    256,
                    4,
                    strides=2,
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
            ]
        ),
        # 5th upsample: 256 -> 128
        tf.keras.Sequential(
            [
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2DTranspose(
                    128,
                    4,
                    strides=2,
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
            ]
        ),
        # 6th upsample: 128 -> 64
        tf.keras.Sequential(
            [
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2DTranspose(
                    64,
                    4,
                    strides=2,
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
            ]
        ),
    ]

    # Final output layer
    last = tf.keras.Sequential(
        [
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(
                CHANNELS,
                4,
                strides=2,
                padding="same",
                kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
                activation="tanh",
            ),
        ]
    )

    # Build the UNet structure with skip connections
    x = inputs

    # Downsampling
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling with skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def export_model(checkpoint_dir):
    """Export the generator model to TensorFlow.js format."""
    print(f"Looking for latest checkpoint in {checkpoint_dir}")
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    if not latest_checkpoint:
        raise ValueError(f"No checkpoint found in {checkpoint_dir}")

    print(f"Found checkpoint: {latest_checkpoint}")

    # Create and initialize the model
    generator = create_generator_model()
    generator(tf.random.normal([1, 256, 256, 3]))  # Build model with dummy input
    generator.summary()

    # Setup checkpoint and attempt to restore
    checkpoint = tf.train.Checkpoint(generator=generator)

    try:
        # Restore weights
        print(f"Restoring from: {latest_checkpoint}")
        checkpoint.restore(latest_checkpoint).expect_partial()
        print("Restore completed successfully")

        # Export model
        export_to_tfjs(generator)

    except Exception as e:
        print(f"Direct restore failed: {e}")
        print("Attempting manual variable extraction...")
        manual_restore(generator, latest_checkpoint)
        export_to_tfjs(generator)


def export_to_tfjs(model):
    """Save model to TensorFlow.js format."""
    export_path = "tmp_saved_model"

    print(f"Saving model to {export_path}")
    tf.saved_model.save(model, export_path)

    print("Converting to TensorFlow.js format...")
    tfjs.converters.convert_tf_saved_model(
        export_path, output_dir="web_model", skip_op_check=True, strip_debug_ops=True
    )

    print("Model exported successfully to 'web_model/'")

    # Cleanup
    if os.path.exists(export_path):
        import shutil

        shutil.rmtree(export_path)
        print(f"Cleaned up temporary directory: {export_path}")


def manual_restore(model, checkpoint_path):
    """Manually restore variables from checkpoint."""
    reader = tf.train.load_checkpoint(checkpoint_path)

    for var in model.trainable_variables:
        var_name = var.name.replace(":0", "")
        checkpoint_name = f"generator/{var_name}"

        try:
            if reader.has_tensor(checkpoint_name):
                tensor_value = reader.get_tensor(checkpoint_name)
                var.assign(tensor_value)
                print(f"Loaded: {var.name}")
            else:
                print(f"Missing: {checkpoint_name}")
        except Exception as ex:
            print(f"Error loading {var.name}: {ex}")


if __name__ == "__main__":
    checkpoint_dir = f"{PATH}/footprints/checkpoint"
    export_model(checkpoint_dir)
