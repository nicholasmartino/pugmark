#!/bin/bash

# Define the project ID
PROJECT_ID="pugmark-448918"  # Replace with your actual project ID

# Set environment variables to ensure all Google Cloud libraries use the same project ID
export GOOGLE_CLOUD_PROJECT=$PROJECT_ID
export CLOUDSDK_CORE_PROJECT=$PROJECT_ID
export GCLOUD_PROJECT=$PROJECT_ID

# Configure gcloud CLI with project ID
gcloud config set project $PROJECT_ID

# install requirements and run training script
pip install --no-deps --upgrade -r requirements.txt
python src/training/train.py