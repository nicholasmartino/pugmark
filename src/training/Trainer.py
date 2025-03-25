import datetime
import os
import time

import gcsfs
import matplotlib.pyplot as plt
import tensorflow as tf
from Discriminator import Discriminator, calculate_discriminator_loss
from Generator import Generator, calculate_generator_loss, generate_images
from Globals import *
from google.cloud import storage
from Loader import load, load_image_test, load_image_train
from Sampler import downsample, upsample


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

# Initialize summary writer
summary_writer = tf.summary.create_file_writer(
    tf.io.gfile.join(log_dir, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
)

# Load datasets
train_files = fs.ls(os.path.join(PATH, "footprints/train"))
test_files = fs.ls(os.path.join(PATH, "footprints/test"))

# Get sample image shape
test_files = tf.io.gfile.glob(f"{PATH}/footprints/test/*.png")
input_image, real_image = load(f"{test_files[0]}")

"""
INPUT PIPELINE
"""

train_dataset = tf.data.Dataset.list_files(f"{PATH}/footprints/train/*.png")
train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(f"{PATH}/footprints/test/*.png")
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(input_image, 0))
print(down_result.shape)

up_model = upsample(3, 4)
up_result = up_model(down_result)
print(up_result.shape)

generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True)

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


discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True)

discriminated = discriminator([input_image[tf.newaxis, ...], generated], training=False)
# Discriminated image can be plotted with matplotlib
plt.imshow(discriminated[0, ..., -1], vmin=-20, vmax=20, cmap="RdBu_r")


for example_input, example_target in test_dataset.take(1):
    generate_images(generator, example_input, example_target)


"""
DEFINE THE OPTIMIZERS AND CHECKPOINT-SAVER
"""

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Create a simple epoch counter for checkpoint restoration
epoch_counter = tf.Variable(0, trainable=False, dtype=tf.int64, name="epoch_counter")

# Create a step counter variable
step_counter = tf.Variable(0, trainable=False, dtype=tf.int64, name="step_counter")

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
    epoch_counter=epoch_counter,
    step_counter=step_counter,
)

"""
TRAINING
"""


@tf.function
def train_step(input_image, target):
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

    return (
        generator_total_loss,
        generator_gan_loss,
        generator_l1_loss,
        discriminator_loss,
    )


def log_metrics(gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss, step):
    """Log training metrics to TensorBoard"""
    with summary_writer.as_default():
        tf.summary.scalar("gen_total_loss", gen_total_loss, step=step)
        tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step)
        tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step)
        tf.summary.scalar("disc_loss", disc_loss, step=step)


def fit(train_ds, test_ds, epochs):
    """Trains the model for the specified number of epochs with checkpoint restoration."""

    # Check for existing checkpoints
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f"Restoring from checkpoint: {latest_checkpoint}")
        checkpoint.restore(latest_checkpoint)
        start_epoch = int(epoch_counter.numpy())
        # Reset step counter if needed
        global step_counter
        step_counter.assign(start_epoch * len(train_ds))
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        step_counter.assign(0)
        print("Starting fresh training")

    for epoch in range(start_epoch, epochs):
        # Update epoch counter for checkpoint restoration
        epoch_counter.assign(epoch)

        start = time.time()

        # Test at the beginning of each epoch
        for example_input, example_target in test_ds.take(1):
            generate_images(generator, example_input, example_target)

        print(f"Epoch {epoch+1}/{epochs}")

        # Training
        for step, (input_image, target) in enumerate(train_ds):
            print(".", end="", flush=True)
            if (step + 1) % 100 == 0:
                print(f"\nStep {step+1}")

            # Run the training step (doesn't need step parameter)
            gen_total, gen_gan, gen_l1, disc = train_step(input_image, target)

            # Update step counter
            current_step = step_counter.assign_add(1)

            # Log metrics separately (doesn't cause retracing)
            log_metrics(gen_total, gen_gan, gen_l1, disc, current_step)

        # Save checkpoint every epoch or at specified intervals
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            # Ensure directory exists before saving
            tf.io.gfile.makedirs(os.path.dirname(checkpoint_prefix))
            checkpoint_path = checkpoint.save(file_prefix=checkpoint_prefix)
            print(f"\nCheckpoint saved at epoch {epoch+1}: {checkpoint_path}")

        print(f"\nTime taken for epoch {epoch+1}: {time.time()-start:.2f} sec")

    print("Training completed")


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

    # Start training
    fit(train_dataset, test_dataset, epochs)
