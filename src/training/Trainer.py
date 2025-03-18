import datetime
import logging
import os
import time
from pathlib import Path

import gcsfs
import tensorflow as tf
from Discriminator import Discriminator, calculate_discriminator_loss
from Generator import Generator, calculate_generator_loss, generate_images
from Globals import *
from google.cloud import storage
from Loader import load, load_image_test, load_image_train
from Sampler import downsample, upsample

start_time = datetime.datetime.now()


# Test Google Cloud Storage access in detail
def test_gcs_access():
    print("\n===== TESTING GCS ACCESS =====")

    # 1. Check authentication method
    print("Checking authentication method...")
    from google.colab import auth

    try:
        # Attempt to authenticate in Colab
        auth.authenticate_user()
        print("✓ Successfully authenticated with Colab")
    except:
        print("Not running in Colab or authentication already done")

    # 2. Check environment variables
    print("\nChecking environment variables...")
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        print(
            f"✓ GOOGLE_APPLICATION_CREDENTIALS is set to: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}"
        )
    else:
        print("✗ GOOGLE_APPLICATION_CREDENTIALS is not set")

    # 3. Test bucket access
    print("\nTesting bucket access...")
    try:
        client = storage.Client()
        bucket = client.get_bucket("metro-vancouver-regional-district")
        print(f"✓ Bucket exists: {bucket.exists()}")

        # 4. Try writing a test file
        print("\nTesting write permissions...")
        test_blob = bucket.blob(
            f"{PATH.replace('gs://metro-vancouver-regional-district/', '')}/test_write_permissions.txt"
        )
        test_blob.upload_from_string("Testing write permissions")
        print(f"✓ Successfully wrote to: {test_blob.name}")

        # 5. Try reading the test file
        print("\nTesting read permissions...")
        content = test_blob.download_as_text()
        print(f"✓ Successfully read content: {content}")

        # 6. Try listing objects
        print("\nTesting list permissions...")
        blobs = list(
            bucket.list_blobs(
                prefix=f"{PATH.replace('gs://metro-vancouver-regional-district/', '')}/",
                max_results=5,
            )
        )
        print(f"✓ Listed {len(blobs)} objects in path")
        for blob in blobs[:5]:  # Show first 5
            print(f"  - {blob.name}")

        # Clean up test file
        test_blob.delete()
        print("✓ Cleaned up test file")

        print("\n✓ ALL GCS PERMISSION TESTS PASSED")
        return True

    except Exception as e:
        print(f"✗ GCS access error: {e}")
        print("\n✗ GCS PERMISSION TESTS FAILED")
        return False


# Run the test
gcs_access_ok = test_gcs_access()
if not gcs_access_ok:
    # If GCS access fails, raise an exception to stop execution
    raise Exception(
        "ERROR: Cannot access Google Cloud Storage. Training cannot proceed."
    )

# Continue with GCS paths
# Verify bucket access
client = storage.Client()
bucket = client.get_bucket("metro-vancouver-regional-district")
print(f"Bucket exists: {bucket.exists()}")

# Define GCS paths
log_dir = f"{PATH}/footprints/logs"
checkpoint_dir = f"{PATH}/footprints/checkpoint"

print(f"Using GCS paths:")
print(f"- Log directory: {log_dir}")
print(f"- Checkpoint directory: {checkpoint_dir}")

# Use explicit GCS file system for directory operations
fs = gcsfs.GCSFileSystem()

# Explicitly create the checkpoint directory in GCS
try:
    if not fs.exists(checkpoint_dir):
        fs.mkdir(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")
    else:
        print(f"Checkpoint directory already exists: {checkpoint_dir}")

    # List contents to verify
    checkpoint_contents = fs.ls(checkpoint_dir)
    print(f"Checkpoint directory contents: {checkpoint_contents}")
except Exception as e:
    print(f"Error accessing checkpoint directory: {e}")
    raise Exception(f"Failed to access or create checkpoint directory: {e}")

# Ensure log directory exists using TensorFlow's file operations
try:
    tf.io.gfile.makedirs(log_dir)
    print(f"Verified log directory: {log_dir}")
except Exception as e:
    print(f"Error creating log directory: {e}")
    raise Exception(f"Failed to create log directory: {e}")

# Initialize summary writer with GCS-compatible path
global summary_writer
summary_writer = tf.summary.create_file_writer(
    tf.io.gfile.join(log_dir, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
)

fs = gcsfs.GCSFileSystem()
train_files = fs.ls(os.path.join(PATH, "footprints/train"))
test_files = fs.ls(os.path.join(PATH, "footprints/test"))

# Update file listing to use full paths
test_files = tf.io.gfile.glob(f"{PATH}/footprints/test/*.png")
input_image, real_image = load(f"{test_files[0]}")
print(input_image.shape)


# The images below are going through random jittering to
# 1. Resize an image to bigger height and width
# 2. Randomly crop to the target size
# 3. Randomly flip the image horizontally


# plt.figure(figsize=(6, 6))
# for i in range(4):
#     rj_inp, rj_re = random_jitter(inp, re)
#     if PLOT:
#         plt.subplot(2, 2, i + 1)
#         plt.imshow(rj_inp / 255.0)
#         plt.axis("off")
# plt.show()


"""
INPUT PIPELINE
"""

# Updated dataset pipeline
train_dataset = tf.data.Dataset.list_files(f"{PATH}/footprints/train/*.png")
train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.repeat()
print(train_dataset.element_spec[0])

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

checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
    epoch_counter=epoch_counter,  # Add epoch counter to the checkpoint
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
):
    # Set up logging to file
    log_dir = Path(checkpoint_directory).parent / "logs"

    # For GCS, use different approach to construct log file path
    if checkpoint_directory.startswith("gs://"):
        log_file_path = (
            f"{checkpoint_directory.rsplit('/', 1)[0]}/logs/training_log.txt"
        )
        print(f"Using GCS log file path: {log_file_path}")

        # Configure logging - for GCS, we'll primarily use print since file handlers may not work with GCS
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
    else:
        # Local file system approach
        os.makedirs(log_dir, exist_ok=True)
        log_file = log_dir / "training_log.txt"
        log_file_path = str(log_file)

        # Configure logging to write to both file and console
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
        )

    # Log whether we're resuming training
    if initial_epoch > 0:
        logging.info(f"Resuming training from epoch {initial_epoch}")
        print(f"Resuming training from epoch {initial_epoch}")
    else:
        logging.info(f"Starting training for {epochs} epochs")
        print(f"Starting training for {epochs} epochs")

    logging.info(f"Checkpoints will be saved to: {checkpoint_directory}")
    print(f"Checkpoints will be saved to: {checkpoint_directory}")
    logging.info(f"Log file location: {log_file_path}")
    print(f"Log file location: {log_file_path}")

    # For GCS, explicitly define checkpoint prefix to ensure it's GCS-compatible
    if checkpoint_directory.startswith("gs://"):
        checkpoint_prefix = f"{checkpoint_directory}/ckpt"
    else:
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    print(f"Checkpoint prefix: {checkpoint_prefix}")

    # Ensure checkpoint directory exists - redundant, but just to be safe
    if checkpoint_directory.startswith("gs://"):
        # GCS directory - already created earlier
        pass
    else:
        # Local directory
        os.makedirs(checkpoint_directory, exist_ok=True)

    start = time.time()

    # Modify the range to start from initial_epoch
    for epoch in range(initial_epoch, epochs):
        # Update epoch counter variable
        epoch_counter.assign(epoch)

        start = time.time()

        logging.info(f"Epoch {epoch+1}/{epochs} - Starting")

        # Test on example batch at start of epoch
        for example_input, example_target in test_dataset.take(1):
            generate_images(generator, example_input, example_target)

        logging.info(f"Epoch: {epoch}")

        # Train
        steps_total = 0
        for n, (input_image, target) in train_dataset.enumerate():
            print(".", end="")
            steps_total = n + 1
            if (n + 1) % 100 == 0:
                print()
                logging.info(f"  - Completed {n+1} steps")
            train_step(input_image, target, epoch)

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(
                f"Attempting to save checkpoint at epoch {epoch+1} to: {checkpoint_prefix}"
            )
            try:
                checkpoint_path = checkpoint.save(file_prefix=checkpoint_prefix)
                logging.info(f"Checkpoint saved at epoch {epoch+1}")
                print(f"Checkpoint saved to: {checkpoint_path}")

                # Verify files were created
                fs = gcsfs.GCSFileSystem()
                checkpoint_files = fs.ls(checkpoint_directory)
                print(f"Files in checkpoint directory after save: {checkpoint_files}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
                logging.error(f"Error saving checkpoint: {e}")
                raise Exception(
                    f"Critical error: Failed to save checkpoint to GCS: {e}"
                )

        epoch_time = time.time() - start
        logging.info(f"Time taken for epoch {epoch+1} is {epoch_time:.2f} sec")
        logging.info(f"Steps completed: {steps_total}")

    # Final checkpoint
    print(f"Saving final checkpoint to: {checkpoint_prefix}")
    try:
        checkpoint_path = checkpoint.save(file_prefix=checkpoint_prefix)
        logging.info(f"Training complete - final checkpoint saved")
        print(f"Final checkpoint saved to: {checkpoint_path}")

        # Verify files were created
        fs = gcsfs.GCSFileSystem()
        checkpoint_files = fs.ls(checkpoint_directory)
        print(f"Files in checkpoint directory after final save: {checkpoint_files}")
    except Exception as e:
        print(f"Error saving final checkpoint: {e}")
        logging.error(f"Error saving final checkpoint: {e}")
        raise Exception(f"Critical error: Failed to save final checkpoint to GCS: {e}")


def train(resume_training=False):
    # Add verification checks
    logging.info(f"TensorFlow version: {tf.__version__}")
    gpu_devices = tf.config.list_physical_devices("GPU")
    logging.info(f"GPU devices: {gpu_devices}")

    # Force GPU placement test
    with tf.device("/GPU:0"):
        test_tensor = tf.random.normal([2, 2])
        logging.info(f"Tensor device test: {test_tensor.device}")

    # Default initial epoch
    initial_epoch = 0

    # Check for existing checkpoint if resuming
    if resume_training:
        # Try to restore the checkpoint
        print(f"Looking for checkpoints in: {checkpoint_dir}")

        # List all files in the checkpoint directory using gcsfs
        try:
            fs = gcsfs.GCSFileSystem()
            checkpoint_files = fs.ls(checkpoint_dir)
            print(f"Files in checkpoint directory: {checkpoint_files}")
        except Exception as e:
            raise Exception(f"Error listing checkpoint directory: {e}")

        # Get the latest checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        print(f"Latest checkpoint found: {latest_checkpoint}")

        if latest_checkpoint:
            try:
                logging.info(f"Restoring from checkpoint: {latest_checkpoint}")
                status = checkpoint.restore(latest_checkpoint)
                # For eager execution
                if hasattr(status, "assert_existing_objects_matched"):
                    status.assert_existing_objects_matched()
                    print("Checkpoint objects matched existing objects")

                # Get the epoch from the restored checkpoint
                initial_epoch = (
                    int(epoch_counter.numpy()) + 1
                )  # +1 because we want to start with the next epoch

                logging.info(
                    f"Checkpoint restored successfully, resuming from epoch {initial_epoch}"
                )
                print(
                    f"Successfully restored checkpoint. Resuming from epoch {initial_epoch}"
                )
            except Exception as e:
                logging.error(f"Error restoring checkpoint: {e}")
                raise Exception(f"Failed to restore checkpoint: {e}")
        else:
            logging.warning("No checkpoint found, starting training from beginning")
            print("No checkpoint found. Starting fresh training.")
            resume_training = False
            initial_epoch = 0
    else:
        logging.warning("No checkpoint found, starting training from beginning")
        print("No checkpoint found. Starting fresh training.")
        resume_training = False
        initial_epoch = 0

    # Rest of original training code
    fit(
        train_dataset,
        test_dataset,
        generator,
        checkpoint,
        checkpoint_dir,
        EPOCHS,
        initial_epoch=initial_epoch,
    )

    # restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    process_time = datetime.datetime.now() - start_time
    logging.info(f"Training finished in {process_time/60} minutes")
