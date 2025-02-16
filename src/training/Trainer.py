import datetime
import os
import time

import gcsfs
import tensorflow as tf
from AuthUtils import auth_client_from_cloud, auth_client_locally
from Discriminator import Discriminator, calculate_discriminator_loss
from Generator import Generator, calculate_generator_loss, generate_images
from Globals import *
from IPython import display
from Loader import load, load_image_test, load_image_train
from Sampler import downsample, upsample

start_time = datetime.datetime.now()

if os.getenv("CLOUD_RUN_JOB"):  # This environment variable is present in Cloud Run
    print("Running in Google Cloud environment")
    client = auth_client_from_cloud()
else:
    print("Running locally")
    client = auth_client_locally()

# Verify bucket access
bucket = client.get_bucket("metro-vancouver-regional-district")
print(f"Bucket exists: {bucket.exists()}")

# Update paths to use GCS when running in cloud
log_dir = tf.io.gfile.join(PATH, "footprints/logs")
checkpoint_dir = tf.io.gfile.join(PATH, "footprints/checkpoint")

# Ensure log directory exists
tf.io.gfile.makedirs(log_dir)

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
):
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    for epoch in range(epochs):
        start = time.time()

        display.clear_output(wait=True)

        for ex_input, ex_target in test_dataset.take(1):
            generate_images(generator, ex_input, ex_target)
        print("Epoch: ", epoch)

        # Train
        for n, (input_image, target) in train_dataset.enumerate():
            print(".", end="")
            if (n + 1) % 100 == 0:
                print()
            train_step(input_image, target, epoch)

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(
            "Time taken for epoch {} is {} sec\n".format(epoch + 1, time.time() - start)
        )
    checkpoint.save(file_prefix=checkpoint_prefix)


def train():
    fit(train_dataset, test_dataset, generator, checkpoint, checkpoint_dir, EPOCHS)

    # restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    process_time = datetime.datetime.now() - start_time
    print(f"Training finished in {process_time/60} minutes")
