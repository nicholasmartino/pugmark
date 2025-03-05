#!/bin/bash

# Check if the notebook instance already exists and delete it if it does
NOTEBOOK_NAME="pugmark-notebook"
NOTEBOOK_EXISTS=$(gcloud notebooks instances list \
  --project $GCP_PROJECT_ID \
  --location us-central1 \
  --filter="name:$NOTEBOOK_NAME" \
  --format="value(name)")

if [ -n "$NOTEBOOK_EXISTS" ]; then
  echo "Deleting existing notebook: $NOTEBOOK_NAME"
  gcloud notebooks instances delete $NOTEBOOK_NAME \
    --project $GCP_PROJECT_ID \
    --location us-central1 \
    --quiet
fi

# Create a new Colab Enterprise notebook on Vertex AI with GPU
gcloud notebooks instances create $NOTEBOOK_NAME \
  --project $GCP_PROJECT_ID \
  --location us-central1 \
  --vm-image-project deeplearning-platform-release \
  --vm-image-family colab-enterprise-gpu \
  --machine-type n1-standard-4 \
  --accelerator-type NVIDIA_TESLA_T4 \
  --accelerator-core-count 1 \
  --boot-disk-size 100GB \
  --service-account github-service-account@$GCP_PROJECT_ID.iam.gserviceaccount.com \
  --metadata="install-nvidia-driver=True" \
  --metadata="proxy-mode=service_account" \
  --metadata="container-repository=us-central1-docker.pkg.dev/$GCP_PROJECT_ID/pugmark/pugmark:latest" \
  --no-register-for-update

echo "Waiting for notebook instance to be created..."
gcloud notebooks instances get $NOTEBOOK_NAME \
  --project $GCP_PROJECT_ID \
  --location us-central1

echo "Notebook instance $NOTEBOOK_NAME created with GPU (NVIDIA T4). You can now access it through the Google Cloud console."
echo "URL: https://console.cloud.google.com/vertex-ai/workbench/notebooks/instances?project=$GCP_PROJECT_ID"
