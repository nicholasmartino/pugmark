import datetime
import logging
import os

import gcsfs
import matplotlib.pyplot as plt
import tensorflow as tf
from google.cloud import storage
from tqdm.notebook import tqdm

# Using absolute imports which work in all execution contexts
from src.training.Discriminator import Discriminator, calculate_discriminator_loss
from src.training.Generator import Generator, calculate_generator_loss, generate_images
from src.training.Globals import *
from src.training.Loader import load, load_image_test, load_image_train
from src.training.Sampler import downsample, upsample

# region LOGGING

# Fix the logging first
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# endregion LOGGING

# region GCS


# Simple GCS configuration
def setup_gcs():
    """Configure GCS access with minimal setup."""
    try:
        client = storage.Client()
        fs = gcsfs.GCSFileSystem(project=client.project)

        # Set up directories
        log_dir = f"{PATH}/footprints/logs"
        checkpoint_dir = f"{PATH}/footprints/checkpoint"

        # Create directories if they don't exist
        for dir_path in [log_dir, checkpoint_dir]:
            if not fs.exists(dir_path):
                fs.mkdir(dir_path)

        return fs, log_dir, checkpoint_dir
    except Exception as e:
        raise Exception(f"Failed to setup GCS: {e}")


# Initialize GCS and get paths
fs, log_dir, checkpoint_dir = setup_gcs()

# endregion GCS

# region DATASET

# Load datasets
train_files = fs.ls(os.path.join(PATH, "footprints/train"))
test_files = fs.ls(os.path.join(PATH, "footprints/test"))

# Get sample image shape
test_files = tf.io.gfile.glob(f"{PATH}/footprints/test/*.png")
input_image, real_image = load(f"{test_files[0]}")

train_dataset = tf.data.Dataset.list_files(f"{PATH}/footprints/train/*.png")
train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.list_files(f"{PATH}/footprints/test/*.png")
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(input_image, 0))
print(down_result.shape)

up_model = upsample(3, 4)
up_result = up_model(down_result)
print(up_result.shape)

# endregion DATASET

# region GENERATOR

generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

generated = generator(input_image[tf.newaxis, ...], training=False)
# Generated image can be plotted with matplotlib
plt.imshow(generated[0, ...])


# Generator loss

# It is a sigmoid cross entropy loss of the generated images and an array of ones. The paper also includes L1 loss
# which is MAE (mean absolute error) between the generated image and the target image. This allows the generated
# image to become structurally similar to the target image. The formula to calculate the total generator loss =
# gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. This value was decided by the authors of the paper.

# Training procedure for the generator

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# endregion GENERATOR

# region DISCRIMINATOR

discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True)

discriminated = discriminator([input_image[tf.newaxis, ...], generated], training=False)
# Discriminated image can be plotted with matplotlib
plt.imshow(discriminated[0, ..., -1], vmin=-20, vmax=20, cmap="RdBu_r")

# endregion DISCRIMINATOR

# region OPTIMIZERS

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# endregion OPTIMIZERS

# region CHECKPOINT

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
)

# endregion CHECKPOINT

# region TEST

for example_input, example_target in test_dataset.take(1):
    generate_images(generator, example_input, example_target)

# endregion TEST

# region TRAINING

# Initialize summary writer with less frequent writing to save GCS costs
summary_writer = tf.summary.create_file_writer(
    tf.io.gfile.join(log_dir, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
)


def log_metrics(gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss, step):
    """Log training metrics to TensorBoard"""
    # Log metrics every 50 steps for better monitoring
    if step % 50 == 0:
        with summary_writer.as_default():
            tf.summary.scalar("gen_total_loss", gen_total_loss, step=step)
            tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step)
            tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step)
            tf.summary.scalar("disc_loss", disc_loss, step=step)

            # Log balance metrics - values below 0.69 (log(2)) indicate one model is winning
            is_gen_winning = tf.cast(gen_gan_loss < 0.69, tf.float32)
            is_disc_winning = tf.cast(disc_loss < 0.69, tf.float32)
            tf.summary.scalar("gen_winning", is_gen_winning, step=step)
            tf.summary.scalar("disc_winning", is_disc_winning, step=step)


@tf.function
def train_step(input_image, target, step):
    """Training step function optimized to minimize retracing"""
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated = generator(input_image, training=True)

        target_score = discriminator([input_image, target], training=True)
        generated_score = discriminator([input_image, generated], training=True)

        generator_total_loss, generator_gan_loss, generator_l1_loss = (
            calculate_generator_loss(generated_score, generated, target, loss_object)
        )
        discriminator_loss = calculate_discriminator_loss(
            target_score, generated_score, loss_object
        )

    generator_gradients = gen_tape.gradient(
        generator_total_loss, generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        discriminator_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    log_metrics(
        generator_total_loss,
        generator_gan_loss,
        generator_l1_loss,
        discriminator_loss,
        step,
    )

    return (
        generator_total_loss,
        generator_gan_loss,
        generator_l1_loss,
        discriminator_loss,
    )


def fit(train_ds, test_ds, steps=40000):
    example_input, example_target = next(iter(test_ds.take(1)))

    # Create progress bar
    progress_bar = tqdm(total=steps, desc="Training", unit="step")

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        # Convert step tensor to Python int
        step_value = int(step.numpy())

        # Train and update metrics
        gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(
            input_image, target, step
        )

        # Save (checkpoint) the model every 5k steps
        if (step_value + 1) % 5000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        progress_bar.update(1)
        print(f"Step: {step_value}")

    progress_bar.close()


def train(epochs=EPOCHS):
    """Main training function."""
    # Set up GPU if available
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")

    # Perform a quick dataset check before training
    print("Checking dataset...")
    # Check that images have expected range (normalized to [-1, 1])
    for input_batch, target_batch in train_dataset.take(1):
        print(f"Input shape: {input_batch.shape}, Target shape: {target_batch.shape}")
        print(
            f"Input min/max: {tf.reduce_min(input_batch):.2f}/{tf.reduce_max(input_batch):.2f}"
        )
        print(
            f"Target min/max: {tf.reduce_min(target_batch):.2f}/{tf.reduce_max(target_batch):.2f}"
        )
        # Check if values are in [-1, 1] range as expected after normalization
        if tf.reduce_max(input_batch) > 1.0 or tf.reduce_min(input_batch) < -1.0:
            print("WARNING: Input images may not be properly normalized to [-1, 1]")
        if tf.reduce_max(target_batch) > 1.0 or tf.reduce_min(target_batch) < -1.0:
            print("WARNING: Target images may not be properly normalized to [-1, 1]")
    print("Dataset check complete")

    # Start training
    fit(train_dataset, test_dataset)


# endregion TRAINING
