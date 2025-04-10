#!/bin/bash

# Define the project ID
PROJECT_ID="pugmark-448918"  # Replace with your actual project ID

# Set environment variables to ensure all Google Cloud libraries use the same project ID
export GOOGLE_CLOUD_PROJECT=$PROJECT_ID
export CLOUDSDK_CORE_PROJECT=$PROJECT_ID
export GCLOUD_PROJECT=$PROJECT_ID

# Configure gcloud CLI with project ID
gcloud config set project $PROJECT_ID

# Navigate to repository root 
cd /content/pugmark

# Create a .env file to tell Python where to find modules
echo "PYTHONPATH=/content/pugmark" > .env

# Install requirements
pip install --no-deps --upgrade -r src/training/requirements.txt

# Make sure python-dotenv is installed for environment variable loading
pip install python-dotenv

# Set PYTHONPATH for this session
export PYTHONPATH=/content/pugmark

# Run the training script as a module
python -m src.training.train