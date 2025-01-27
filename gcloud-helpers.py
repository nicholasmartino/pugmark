import os

import tensorflow as tf
from google.cloud import storage

# Cloud storage configuration
BUCKET_NAME = "metro-vancouver-regional-district"
MODEL_DIR = "models"
TRAINING_DATA_DIR = "training_data"


def download_from_gcs(bucket_name, source_blob_path, destination_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_path)
    blob.download_to_filename(destination_path)


def upload_to_gcs(bucket_name, source_path, destination_blob_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_path)
    blob.upload_from_filename(source_path)
