import datetime
import logging
import os
import time

import gcsfs
import matplotlib.pyplot as plt
import tensorflow as tf
from google.cloud import storage

# Using absolute imports which work in all execution contexts
from src.training.Discriminator import Discriminator, calculate_discriminator_loss
from src.training.Generator import (
    Generator,
    calculate_generator_loss,
    generate_detailed_check_plots,
    generate_images,
)
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

# endregion GENERATOR

# region DISCRIMINATOR

discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True)

discriminated = discriminator([input_image[tf.newaxis, ...], generated], training=False)
# Discriminated image can be plotted with matplotlib
plt.imshow(discriminated[0, ..., -1], vmin=-20, vmax=20, cmap="RdBu_r")

# endregion DISCRIMINATOR

# region TEST

for example_input, example_target in test_dataset.take(1):
    generate_images(generator, example_input, example_target)

# endregion TEST

# region OPTIMIZERS

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# endregion OPTIMIZERS

# region CHECKPOINT

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

# Create a checkpoint manager to limit the number of checkpoints
# This helps reduce GCS storage costs by keeping only the latest N checkpoints
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint,
    checkpoint_dir,
    max_to_keep=MAX_CHECKPOINTS_TO_KEEP,
    checkpoint_name="ckpt",
)

# endregion CHECKPOINT

# region TRAINING

# Initialize summary writer with less frequent writing to save GCS costs
summary_writer = tf.summary.create_file_writer(
    tf.io.gfile.join(log_dir, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
)


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


def fit(train_ds, test_ds, epochs):
    """Trains the model for the specified number of epochs with checkpoint restoration."""
    # Count training files for progress tracking
    train_files = tf.io.gfile.glob(f"{PATH}/footprints/train/*.png")
    total_steps = len(train_files) // BATCH_SIZE
    print(f"Total steps per epoch: {total_steps}")

    # Check for existing checkpoints
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f"Restoring from checkpoint: {latest_checkpoint}")
        checkpoint.restore(latest_checkpoint)
        start_epoch = int(epoch_counter.numpy())
        current_global_step = int(step_counter.numpy())
        epoch_step_offset = current_global_step - (start_epoch * total_steps)
        print(
            f"Resuming training from epoch {start_epoch + 1} at step {current_global_step} (epoch step: {epoch_step_offset})"
        )
    else:
        start_epoch = 0
        step_counter.assign(0)
        epoch_step_offset = 0
        print("Starting fresh training")

    # Calculate best checkpoint based on validation metrics
    best_gen_loss = float("inf")

    # Create a prediction history buffer for visualization
    prediction_history = []
    check_example_input = None
    check_example_target = None

    # Get a single example for consistent check plots
    for example_input, example_target in test_ds.take(1):
        check_example_input = example_input
        check_example_target = example_target
        # Generate initial detailed check plot
        if PLOT:
            print("Generating initial check plot...")
            generate_detailed_check_plots(
                generator,
                check_example_input,
                check_example_target,
                prediction_history,
                0,
            )

    for epoch in range(start_epoch, epochs):
        # Update epoch counter for checkpoint restoration
        epoch_counter.assign(epoch)
        start = time.time()

        # Test at the beginning of each epoch - only generate detailed check plots every 5 epochs to save GCS costs
        if epoch % 5 == 0 and check_example_input is not None:
            print(f"Generating detailed check plot for epoch {epoch+1}...")
            prediction = generate_detailed_check_plots(
                generator,
                check_example_input,
                check_example_target,
                prediction_history,
                int(step_counter.numpy()),
            )
            # Store the prediction for history tracking (only keep the most recent 20)
            if len(prediction_history) >= 20:
                prediction_history.pop(0)  # Remove oldest
            prediction_history.append(prediction)
        # For other epochs, just generate a simple plot
        elif check_example_input is not None:
            generate_images(generator, check_example_input, check_example_target)

        print(f"Epoch {epoch+1}/{epochs}")

        # Skip already processed steps in the first epoch after restoration
        if epoch == start_epoch and epoch_step_offset > 0:
            train_ds_epoch = train_ds.skip(epoch_step_offset)
            print(f"Skipping {epoch_step_offset} already processed steps...")
        else:
            train_ds_epoch = train_ds

        # Training loop with simple progress bar
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        step_count = 0
        gen_wins = 0
        disc_wins = 0

        for input_image, target in train_ds_epoch:
            # Run training step
            gen_total, gen_gan, gen_l1, disc = train_step(input_image, target)
            epoch_gen_loss += gen_total
            epoch_disc_loss += disc
            step_count += 1

            # Track win counts for balance monitoring
            if gen_gan < 0.69:  # log(2) indicates generator is winning
                gen_wins += 1
            if disc < 0.69:  # log(2) indicates discriminator is winning
                disc_wins += 1

            current_step = step_counter.assign_add(1)
            step_number = current_step.numpy()
            log_metrics(gen_total, gen_gan, gen_l1, disc, current_step)

            # Simple progress bar that's resistant to interruptions
            progress = min(1.0, (step_number + 1) / total_steps)
            bar_width = 30
            bar = "█" * int(bar_width * progress) + "░" * (
                bar_width - int(bar_width * progress)
            )

            # Calculate ETA
            elapsed = time.time() - start
            if step_number > 0:
                remaining_steps = total_steps - (step_number + 1)
                eta = elapsed * (remaining_steps / (step_number + 1))
                eta_str = f"ETA: {eta:.1f}s"
            else:
                eta_str = "ETA: --"

            # Force clear entire line before printing progress
            print(f"\r{' ' * 120}", end="")
            print(
                f"\r[{bar}] {progress*100:3.0f}% | {step_number+1}/{total_steps} | {eta_str} | Global step: {current_step}",
                end="",
            )

            if step_number + 1 >= total_steps:
                break

        # Calculate average loss for the epoch
        avg_gen_loss = epoch_gen_loss / step_count if step_count > 0 else float("inf")
        avg_disc_loss = epoch_disc_loss / step_count if step_count > 0 else float("inf")
        win_ratio = gen_wins / max(1, (gen_wins + disc_wins))

        print(
            f"\nEpoch {epoch+1}: {time.time()-start:.1f}s | Avg Gen Loss: {avg_gen_loss:.4f} | Avg Disc Loss: {avg_disc_loss:.4f}"
        )
        print(f"Gen/Disc balance: {gen_wins}/{disc_wins} wins (ratio: {win_ratio:.2f})")

        # Save checkpoint at the end of each epoch
        checkpoint_path = checkpoint_manager.save()
        print(f"Saved checkpoint to {checkpoint_path}")

        # Also save a special checkpoint if this is the best model so far
        if avg_gen_loss < best_gen_loss:
            best_gen_loss = avg_gen_loss
            # Save with a special name for best model
            special_checkpoint_path = os.path.join(checkpoint_dir, "best_model")
            checkpoint.write(special_checkpoint_path)
            print(f"New best model saved: {special_checkpoint_path}")

    print("Done!")


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
    fit(train_dataset, test_dataset, epochs)


# endregion TRAINING
