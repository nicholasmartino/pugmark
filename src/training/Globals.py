# Keep the original GCS path
SECRETS_PATH = "secrets/pugmark-448918-fc1f96c413b4.json"
PATH = "gs://metro-vancouver-regional-district/processed"
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
CHANNELS = 3
PLOT = False
EPOCHS = 300
LAMBDA = 100

# GCS storage optimization settings
MAX_CHECKPOINTS_TO_KEEP = 5
CHECKPOINT_SAVE_FREQ = 5
CHECKPOINT_STEP_FREQ = 1000
