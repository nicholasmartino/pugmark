#!/bin/bash

# Set -e to make the script exit on error
set -e

# Check if GCP_PROJECT_ID is set
if [ -z "$GCP_PROJECT_ID" ]; then
  echo "Error: GCP_PROJECT_ID environment variable is not set"
  echo "Please set it with: export GCP_PROJECT_ID=your-project-id"
  exit 1
fi

# Define an array of zones to try (removing us-central1-f which had permission issues)
ZONES=("us-central1-a" "us-central1-b" "us-central1-c" "us-west1-b" "us-east1-c")
NOTEBOOK_NAME="pugmark-notebook"

# Define GPU configurations to try in order of preference
GPU_CONFIGS=(
  # Original configurations - T4 GPUs
  "machine-type=n1-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-core-count=1"
  "machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-core-count=1"
  
  # P4 GPUs
  "machine-type=n1-standard-4,accelerator-type=NVIDIA_TESLA_P4,accelerator-core-count=1"
  "machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_P4,accelerator-core-count=1"
  
  # N2 machine types with T4 GPUs
  "machine-type=n2-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-core-count=1"
  "machine-type=n2-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-core-count=1"
  
  # Compute-optimized with T4 GPUs
  "machine-type=c2-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-core-count=1"
  "machine-type=c2-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-core-count=1"
  
  # More powerful GPUs
  "machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_P100,accelerator-core-count=1"
  "machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_V100,accelerator-core-count=1"
  
  # Higher-end configurations
  "machine-type=n1-highmem-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-core-count=1"
  "machine-type=n1-highmem-8,accelerator-type=NVIDIA_TESLA_P4,accelerator-core-count=1"
  
  # Latest gen A100 GPUs (if available in your project)
  "machine-type=a2-highgpu-1g,accelerator-type=NVIDIA_TESLA_A100,accelerator-core-count=1"
)

# Try different zones and GPU configurations
for ZONE in "${ZONES[@]}"; do
  # Check if the notebook instance already exists in this zone and delete it
  NOTEBOOK_EXISTS=$(gcloud notebooks instances list \
    --project $GCP_PROJECT_ID \
    --location $ZONE \
    --filter="name:$NOTEBOOK_NAME" \
    --format="value(name)" 2>/dev/null || echo "")

  if [ -n "$NOTEBOOK_EXISTS" ]; then
    echo "Deleting existing notebook: $NOTEBOOK_NAME in zone $ZONE"
    gcloud notebooks instances delete $NOTEBOOK_NAME \
      --project $GCP_PROJECT_ID \
      --location $ZONE \
      --quiet
  fi
  
  # Try each GPU configuration
  for CONFIG in "${GPU_CONFIGS[@]}"; do
    # Parse the configuration
    MACHINE_TYPE=$(echo $CONFIG | cut -d',' -f1 | cut -d'=' -f2)
    ACCELERATOR_TYPE=$(echo $CONFIG | cut -d',' -f2 | cut -d'=' -f2)
    ACCELERATOR_COUNT=$(echo $CONFIG | cut -d',' -f3 | cut -d'=' -f2)
    
    echo "Attempting to create notebook in zone: $ZONE with $ACCELERATOR_TYPE GPU on $MACHINE_TYPE"
    echo "Creating new notebook instance: $NOTEBOOK_NAME in zone $ZONE"
    
    # Create a new Notebook instance on Vertex AI with GPU
    if gcloud notebooks instances create $NOTEBOOK_NAME \
      --project $GCP_PROJECT_ID \
      --location $ZONE \
      --container-repository=us-central1-docker.pkg.dev/$GCP_PROJECT_ID/pugmark/pugmark \
      --container-tag=latest \
      --machine-type $MACHINE_TYPE \
      --accelerator-type $ACCELERATOR_TYPE \
      --accelerator-core-count $ACCELERATOR_COUNT \
      --boot-disk-size 100 \
      --service-account github-service-account@$GCP_PROJECT_ID.iam.gserviceaccount.com \
      --metadata="install-nvidia-driver=True" \
      --metadata="proxy-mode=service_account"; then
      
      echo "Successfully created notebook in zone: $ZONE with $ACCELERATOR_TYPE GPU on $MACHINE_TYPE"
      
      # Try to describe the instance to verify it exists
      if gcloud notebooks instances describe $NOTEBOOK_NAME \
        --project $GCP_PROJECT_ID \
        --location $ZONE; then
        echo "Notebook instance $NOTEBOOK_NAME created successfully with $ACCELERATOR_TYPE GPU on $MACHINE_TYPE in zone $ZONE."
        echo "You can access it through the Google Cloud console:"
        echo "URL: https://console.cloud.google.com/vertex-ai/workbench/notebooks/instances?project=$GCP_PROJECT_ID"
        exit 0  # Exit with success
      fi
      
      # If we reach here, verification failed
      echo "Notebook creation seemed successful but verification failed. Continuing with next configuration."
    else
      echo "Failed to create notebook in zone $ZONE with $ACCELERATOR_TYPE GPU on $MACHINE_TYPE. Trying next configuration..."
    fi
  done
done

# If we get here, all zones and configurations failed
echo "Failed to create notebook instance in any of the tried zones or configurations."
echo "All zones are currently lacking resources or have other issues."
echo "Please try again later or consider modifying the resource requirements."
exit 1
