# Keep the original GCS path
SECRETS_PATH = "secrets/pugmark-448918-fc1f96c413b4.json"
PATH = "gs://metro-vancouver-regional-district/processed"
BUFFER_SIZE = 1000
BATCH_SIZE = 4
IMG_WIDTH = 256
IMG_HEIGHT = 256
CHANNELS = 3
PLOT = True
LAMBDA = 100
STEPS = 40000

# GCS storage optimization settings
MAX_CHECKPOINTS_TO_KEEP = 5
