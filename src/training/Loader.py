import tensorflow as tf

from src.training.Globals import *
from src.training.Helpers import normalize, random_jitter, resize

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


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image
