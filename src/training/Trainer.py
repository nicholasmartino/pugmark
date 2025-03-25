import datetime
import os
import time

import gcsfs
import tensorflow as tf
from Discriminator import Discriminator, calculate_discriminator_loss
from Generator import Generator, calculate_generator_loss, generate_images
from Globals import *
from google.cloud import storage
from Loader import load, load_image_test, load_image_train
from Sampler import downsample, upsample


# Simple GCS configuration
def setup_gcs():
    try:
        client = storage.Client()
        bucket = client.get_bucket("metro-vancouver-regional-district")
        fs = gcsfs.GCSFileSystem(project=client.project)

        # Set up directories
        log_dir = f"{PATH}/footprints/logs"
        checkpoint_dir = f"{PATH}/footprints/checkpoint"

        # Create directories if they don't exist
        for dir_path in [log_dir, checkpoint_dir]:
            if not fs.exists(dir_path):
                fs.mkdir(dir_path)

        return client.project, fs, log_dir, checkpoint_dir
    except Exception as e:
        raise Exception(f"Failed to setup GCS: {e}")


# Initialize GCS and get paths
project_id, fs, log_dir, checkpoint_dir = setup_gcs()

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
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

generated = generator(input_image[tf.newaxis, ...], training=False)
# Generated image can be plotted with matplotlib
# plt.imshow(generated[0, ...])


# Generator loss

# It is a sigmoid cross entropy loss of the generated images and an array of ones. The paper also includes L1 loss
# which is MAE (mean absolute error) between the generated image and the target image. This allows the generated
# image to become structurally similar to the target image. The formula to calculate the total generator loss =
# gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. This value was decided by the authors of the paper.

# Training procedure for the generator

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

discriminated = discriminator([input_image[tf.newaxis, ...], generated], training=False)
# Discriminated image can be plotted with matplotlib
# plt.imshow(discriminated[0, ..., -1], vmin=-20, vmax=20, cmap="RdBu_r")


for example_input, example_target in test_dataset.take(1):
    generate_images(generator, example_input, example_target)


"""
DEFINE THE OPTIMIZERS AND CHECKPOINT-SAVER
"""

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Track training epoch in a TensorFlow variable so it gets saved in the checkpoint
epoch_counter = tf.Variable(0, trainable=False, dtype=tf.int64, name="epoch_counter")
# Add a step counter to track step within epoch
step_counter = tf.Variable(0, trainable=False, dtype=tf.int64, name="step_counter")

checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
)

"""
TRAINING
* For each example input generate an output.
* The discriminator receives the input_image and the generated image as the first input. The second input is the 
input_image and the target_image.
* Next, we calculate the generator and the discriminator loss.
* Then, we calculate the gradients of loss with respect to both the generator and the discriminator variables(inputs) 
and apply those to the optimizer.
* Then log the losses to TensorBoard.
"""


@tf.function
def train_step(input_image, target, epoch):
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

    with summary_writer.as_default():
        tf.summary.scalar("gen_total_loss", generator_total_loss, step=epoch)
        tf.summary.scalar("gen_gan_loss", generator_gan_loss, step=epoch)
        tf.summary.scalar("gen_l1_loss", generator_l1_loss, step=epoch)
        tf.summary.scalar("disc_loss", discriminator_loss, step=epoch)


"""
The actual training loop:

* Iterates over the number of epochs.
* On each epoch it clears the display, and runs generate_images to show it's progress.
* On each epoch it iterates over the training dataset, printing a '.' for each example.
* It saves a checkpoint every 20 epochs.
"""


def fit(
    train_dataset,
    test_dataset,
    generator,
    checkpoint,
    checkpoint_directory,
    epochs,
    initial_epoch=0,
    initial_step=0,  # Add initial step parameter
):
    # Set up log file path (for reference only now)
    log_dir = f"{checkpoint_directory.rsplit('/', 1)[0]}/logs"
    log_file_path = f"{log_dir}/training_log.txt"
    print(f"Using GCS log file path: {log_file_path}")

    # Remove logging configuration - we'll only use print statements

    # Log whether we're resuming training
    if initial_epoch > 0:
        print(f"Resuming training from epoch {initial_epoch}")
    else:
        print(f"Starting training for {epochs} epochs")

    print(f"Checkpoints will be saved to: {checkpoint_directory}")
    print(f"Log file location: {log_file_path}")

    # Define checkpoint prefix for GCS
    checkpoint_prefix = f"{checkpoint_directory}/ckpt"

    # Ensure the directory marker exists in the bucket
    bucket_name = checkpoint_directory.split("/")[2]
    blob_prefix = "/".join(checkpoint_directory.split("/")[3:])

    client = storage.Client(project=project_id)
    bucket = client.get_bucket(bucket_name)
    bucket.blob(f"{blob_prefix}/").upload_from_string("")
    print(f"Ensured GCS checkpoint directory marker exists: {blob_prefix}/")

    print(f"Checkpoint prefix: {checkpoint_prefix}")

    start = time.time()

    # Modify the range to start from initial_epoch
    for epoch in range(initial_epoch, epochs):
        # Update epoch counter variable
        epoch_counter.assign(epoch)

        start = time.time()

        print(f"Epoch {epoch}/{epochs} - Starting")

        # Test on example batch at start of epoch
        for example_input, example_target in test_dataset.take(1):
            generate_images(generator, example_input, example_target)

        print(f"Epoch: {epoch}")

        # Train
        steps_total = 0

        # When resuming from a checkpoint in the middle of an epoch, skip to the appropriate step
        skip_steps = initial_step if epoch == initial_epoch else 0
        if skip_steps > 0:
            print(f"Resuming from step {skip_steps} in epoch {epoch}")

        # Use dataset.skip() to move to the right position if resuming
        epoch_dataset = train_dataset
        if skip_steps > 0:
            epoch_dataset = epoch_dataset.skip(skip_steps)

        for n, (input_image, target) in epoch_dataset.enumerate():
            # Calculate actual step number (n is 0-based from enumerate)
            actual_step = n + skip_steps + 1
            steps_total = actual_step

            # Update step counter
            step_counter.assign(steps_total)

            print(".", end="")
            if actual_step % 100 == 0:
                print()
                print(f"  - Completed {actual_step} steps")

                # Save checkpoint every 100 steps for more frequent checkpoints
                print(f"Saving checkpoint at step {steps_total} (epoch {epoch})")
                try:
                    # Force a directory check/creation before saving
                    tf.io.gfile.makedirs(os.path.dirname(checkpoint_prefix))

                    step_checkpoint_prefix = (
                        f"{checkpoint_prefix}_ep{epoch}_step{steps_total}"
                    )
                    checkpoint_path = checkpoint.save(
                        file_prefix=step_checkpoint_prefix
                    )
                    print(f"Step checkpoint saved at epoch {epoch}, step {steps_total}")
                    print(f"Step checkpoint saved to: {checkpoint_path}")
                except Exception as e:
                    print(f"Warning: Error saving step checkpoint: {e}")

            train_step(input_image, target, epoch)

        # At the end of an epoch, reset initial_step for the next epoch
        initial_step = 0

        # Always save checkpoint at the end of each epoch
        print(f"Saving checkpoint at end of epoch {epoch} to: {checkpoint_prefix}")
        try:
            checkpoint_path = checkpoint.save(file_prefix=checkpoint_prefix)
            print(f"Epoch checkpoint saved at epoch {epoch}")
            print(f"Epoch checkpoint saved to: {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Error saving epoch checkpoint: {e}")

        epoch_time = time.time() - start
        print(f"Time taken for epoch {epoch} is {epoch_time:.2f} sec")
        print(f"Steps completed: {steps_total}")

    # Final checkpoint
    print(f"Saving final checkpoint to: {checkpoint_prefix}")
    try:
        # Force a directory check/creation before saving
        tf.io.gfile.makedirs(os.path.dirname(checkpoint_prefix))
        print(
            f"Verified GCS directory exists for final checkpoint: {os.path.dirname(checkpoint_prefix)}"
        )

        checkpoint_path = checkpoint.save(file_prefix=checkpoint_prefix)
        print(f"Training complete - final checkpoint saved")
        print(f"Final checkpoint saved to: {checkpoint_path}")

        # Verify files were created
        bucket_name = checkpoint_directory.split("/")[2]
        blob_path = "/".join(checkpoint_directory.split("/")[3:])

        client = storage.Client(project=project_id)
        bucket = client.get_bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=blob_path, max_results=10))
        print(
            f"Files in checkpoint directory after final save: {[b.name for b in blobs]}"
        )
    except Exception as e:
        print(f"Error saving final checkpoint: {e}")
        raise Exception(f"Critical error: Failed to save final checkpoint to GCS: {e}")


def train(resume_training=False):
    # Basic GPU setup
    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    # Initialize or restore from checkpoint
    checkpoint_prefix = f"{checkpoint_dir}/ckpt"
    if resume_training:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            initial_epoch = int(epoch_counter.numpy())
            initial_step = int(step_counter.numpy())
            print(f"Resuming from epoch {initial_epoch}, step {initial_step}")
        else:
            initial_epoch = initial_step = 0
            print("No checkpoint found, starting fresh training")
    else:
        initial_epoch = initial_step = 0

    # Training loop
    for epoch in range(initial_epoch, EPOCHS):
        epoch_counter.assign(epoch)
        start = time.time()

        # Test on example batch
        for example_input, example_target in test_dataset.take(1):
            generate_images(generator, example_input, example_target)

        # Train
        steps_total = 0
        epoch_dataset = (
            train_dataset.skip(initial_step)
            if initial_epoch == epoch and initial_step > 0
            else train_dataset
        )

        for n, (input_image, target) in epoch_dataset.enumerate():
            actual_step = n + initial_step + 1
            step_counter.assign(actual_step)

            train_step(input_image, target, epoch)

            if actual_step % 100 == 0:
                print(f"\nCompleted {actual_step} steps")
                checkpoint.save(
                    file_prefix=f"{checkpoint_prefix}_ep{epoch}_step{actual_step}"
                )

        # Reset initial_step for next epoch
        initial_step = 0

        # Save epoch checkpoint
        checkpoint.save(file_prefix=checkpoint_prefix)
        print(f"\nEpoch {epoch} completed in {time.time() - start:.2f} sec")

    print(f"Training completed in {datetime.datetime.now() - start_time}")
