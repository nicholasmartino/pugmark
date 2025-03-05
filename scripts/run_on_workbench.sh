#!/bin/bash
set -e

# Get parameters from environment variables or use defaults
PROJECT_ID=${GCP_PROJECT_ID:?"GCP_PROJECT_ID is required"}
REGION=${REGION:-"us-central1"}
ZONE=${ZONE:-"us-central1-a"}
INSTANCE_NAME=${WORKBENCH_INSTANCE:?"WORKBENCH_INSTANCE is required"}
CREATE_INSTANCE=${CREATE_INSTANCE:-"true"}
MACHINE_TYPE=${MACHINE_TYPE:-"n1-standard-8"}
ACCELERATOR_TYPE=${ACCELERATOR_TYPE:-"NVIDIA_TESLA_T4"}
ACCELERATOR_COUNT=${ACCELERATOR_COUNT:-"1"}
DELETE_AFTER_TRAINING=${DELETE_AFTER_TRAINING:-"false"}
OUTPUT_BUCKET=${OUTPUT_BUCKET:-"$PROJECT_ID-ml"}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-"pugmark/training"}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-1}
REPO_URL=${GITHUB_SERVER_URL:+"$GITHUB_SERVER_URL/$GITHUB_REPOSITORY"}

echo "=== Running Training on Vertex AI Workbench ==="
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Zone: $ZONE"
echo "Instance Name: $INSTANCE_NAME"
echo "Create Instance: $CREATE_INSTANCE"
echo "Machine Type: $MACHINE_TYPE"
echo "Accelerator Type: $ACCELERATOR_TYPE"
echo "Accelerator Count: $ACCELERATOR_COUNT"
echo "Delete After Training: $DELETE_AFTER_TRAINING"
echo "Output Bucket: $OUTPUT_BUCKET"
echo "Output Prefix: $OUTPUT_PREFIX"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Repository URL: $REPO_URL"
echo "==========================================="

# Install required Python packages
pip install google-cloud-aiplatform google-cloud-storage google-cloud-notebooks

# Execute the Python script
python scripts/vertex_workbench_execute.py \
  --project-id "$PROJECT_ID" \
  --region "$REGION" \
  --zone "$ZONE" \
  --instance-name "$INSTANCE_NAME" \
  --create-instance "$CREATE_INSTANCE" \
  --machine-type "$MACHINE_TYPE" \
  --accelerator-type "$ACCELERATOR_TYPE" \
  --accelerator-count "$ACCELERATOR_COUNT" \
  --delete-after-training "$DELETE_AFTER_TRAINING" \
  --output-bucket "$OUTPUT_BUCKET" \
  --output-prefix "$OUTPUT_PREFIX" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --repo-url "$REPO_URL" 