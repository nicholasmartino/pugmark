import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices())

from Trainer import train

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"An error occurred: {e}")
