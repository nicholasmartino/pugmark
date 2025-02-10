import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices())


if __name__ == "__main__":
    try:
        from Trainer import train

        train()
    except Exception as e:
        print(f"An error occurred: {e}")
