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

# Install only the minimal necessary dependencies for controlling Workbench
pip install google-cloud-aiplatform google-api-python-client google-auth google-auth-httplib2

# Create startup script to run on Workbench instance
cat > setup_and_train.sh << 'EOL'
#!/bin/bash
set -e

# Clone the repository
if [ -n "$REPO_URL" ]; then
  echo "Cloning repository $REPO_URL..."
  rm -rf /tmp/pugmark
  git clone $REPO_URL /tmp/pugmark
  cd /tmp/pugmark
  pip install -r requirements.txt
fi

# Set environment variables
export OUTPUT_DIR="$OUTPUT_PATH"
export AIP_MODE="training"
export TF_FORCE_GPU_ALLOW_GROWTH="true"
export EPOCHS="$EPOCHS"
export BATCH_SIZE="$BATCH_SIZE"

# Run training
echo "Starting training..."
cd /tmp/pugmark
python -c "from src.training.Trainer import train; train()"
EOL

# Use gcloud to create instance (if needed) and execute training
if [ "$CREATE_INSTANCE" = "true" ]; then
  echo "Creating Workbench instance..."
  gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator=type=$ACCELERATOR_TYPE,count=$ACCELERATOR_COUNT \
    --image-family=tf-ent-2-12-cu113-notebooks \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --metadata=install-nvidia-driver=True
fi

# Copy and execute startup script
echo "Copying and executing startup script..."
gcloud compute scp setup_and_train.sh $INSTANCE_NAME:~/setup_and_train.sh --zone=$ZONE
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command "bash ~/setup_and_train.sh" \
  --environment-variables="REPO_URL=$REPO_URL,OUTPUT_PATH=$OUTPUT_BUCKET/$OUTPUT_PREFIX,EPOCHS=$EPOCHS,BATCH_SIZE=$BATCH_SIZE"

# Delete instance if requested
if [ "$CREATE_INSTANCE" = "true" ] && [ "$DELETE_AFTER_TRAINING" = "true" ]; then
  echo "Deleting Workbench instance..."
  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
fi 