import tensorflow as tf
from Globals import CHANNELS, IMG_HEIGHT, IMG_WIDTH


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(
        input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    real_image = tf.image.resize(
        real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
    )
    return cropped_image[0], cropped_image[1]


# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


"""
LOAD THE DATASET

In random jittering, the image is resized to 286 x 286 and then randomly cropped to 256 x 256
In random mirroring, the image is randomly flipped horizontally i.e left to right.
"""


def load(image_file):
    image = tf.io.read_file(image_file)

    # Only for jpeg
    # image = tf.image.decode_jpeg(image)

    # Custom edit for png
    image = tf.io.decode_png(image, channels=CHANNELS)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image
