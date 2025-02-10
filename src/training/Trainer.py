import argparse
import datetime
import os
import time

import gcsfs
import tensorflow as tf
from Discriminator import Discriminator, discriminator_loss
from Generator import Generator, generate_images, generator_loss
from Globals import *
from google.cloud import storage
from Helpers import load, random_jitter
from IPython import display
from Loader import load_image_test, load_image_train
from matplotlib import pyplot as plt
from Sampler import downsample, upsample

print(tf.__version__)
print(tf.config.list_physical_devices())

start_time = datetime.datetime.now()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# _URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
# path_to_zip = tf.keras.utils.get_file('facades.tar.gz', origin=_URL, extract=True)
# PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')

# Configure GCS access (choose one method below)
# Option 1: If running locally, set credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SECRETS_PATH

# Add this before any GCS operations
client = storage.Client.from_service_account_json(SECRETS_PATH)

# Verify bucket access
bucket = client.get_bucket("metro-vancouver-regional-district")
print(f"Bucket exists: {bucket.exists()}")

fs = gcsfs.GCSFileSystem()
train_files = fs.ls(os.path.join(PATH, "train"))
test_files = fs.ls(os.path.join(PATH, "test"))

# Update file listing to use full paths
test_files = tf.io.gfile.glob(f"{PATH}/test/*.png")

inp, re = load(f"{test_files[0]}")
print(inp.shape)

# Casting to int for matplotlib to show the image
if PLOT:
    plt.figure()
    plt.imshow(inp / 255.0)
    plt.figure()
    plt.imshow(re / 255.0)


# The images below are going through random jittering to
# 1. Resize an image to bigger height and width
# 2. Randomly crop to the target size
# 3. Randomly flip the image horizontally


if PLOT:
    plt.figure(figsize=(6, 6))
for i in range(4):
    rj_inp, rj_re = random_jitter(inp, re)
    if PLOT:
        plt.subplot(2, 2, i + 1)
        plt.imshow(rj_inp / 255.0)
        plt.axis("off")
if PLOT:
    plt.show()


"""
INPUT PIPELINE
"""

# Updated dataset pipeline
dataset = tf.data.Dataset.list_files(f"{PATH}/train/*.png")
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE)
print(dataset.element_spec[0])

test_dataset = tf.data.Dataset.list_files(f"{PATH}/test/*.png")
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print(down_result.shape)


up_model = upsample(3, 4)
up_result = up_model(down_result)
print(up_result.shape)

generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

gen_output = generator(inp[tf.newaxis, ...], training=False)
if PLOT:
    plt.imshow(gen_output[0, ...])


# Generator loss

# It is a sigmoid cross entropy loss of the generated images and an array of ones. The paper also includes L1 loss
# which is MAE (mean absolute error) between the generated image and the target image. This allows the generated
# image to become structurally similar to the target image. The formula to calculate the total generator loss =
# gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. This value was decided by the authors of the paper.

# Training procedure for the generator

LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
if PLOT:
    plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap="RdBu_r")


for example_input, example_target in test_dataset.take(1):
    generate_images(generator, example_input, example_target)


log_dir = "logs/"
summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)


"""
DEFINE THE OPTIMIZERS AND CHECKPOINT-SAVER
"""

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = "data/ckpt"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
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
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target, loss_object
        )
        disc_loss = discriminator_loss(
            disc_real_output, disc_generated_output, loss_object
        )

    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    with summary_writer.as_default():
        tf.summary.scalar("gen_total_loss", gen_total_loss, step=epoch)
        tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=epoch)
        tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=epoch)
        tf.summary.scalar("disc_loss", disc_loss, step=epoch)


"""
The actual training loop:

* Iterates over the number of epochs.
* On each epoch it clears the display, and runs generate_images to show it's progress.
* On each epoch it iterates over the training dataset, printing a '.' for each example.
* It saves a checkpoint every 20 epochs.
"""


def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()

        display.clear_output(wait=True)

        for ex_input, ex_target in test_ds.take(1):
            generate_images(generator, ex_input, ex_target)
        print("Epoch: ", epoch)

        # Train
        for n, (input_image, target) in train_ds.enumerate():
            print(".", end="")
            if (n + 1) % 100 == 0:
                print()
            train_step(input_image, target, epoch)
        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(
            "Time taken for epoch {} is {} sec\n".format(epoch + 1, time.time() - start)
        )
    checkpoint.save(file_prefix=checkpoint_prefix)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=str, default="./")
    args = parser.parse_args()

    # Update paths to use GCS when running in cloud
    if args.job_dir.startswith("gs://"):
        log_dir = os.path.join(args.job_dir, "logs")
        checkpoint_dir = os.path.join(args.job_dir, "ckpt")
    else:
        log_dir = "logs"
        checkpoint_dir = "data/ckpt"

    fit(dataset, EPOCHS, test_dataset)

    # restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    process_time = datetime.datetime.now() - start_time
    print(f"Training finished in {process_time/60} minutes")
