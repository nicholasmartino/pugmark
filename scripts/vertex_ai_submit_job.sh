#!/bin/bash
set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID}
REGION="us-central1"
MACHINE_TYPE=${MACHINE_TYPE:-"n1-standard-8"}
ACCELERATOR_TYPE=${ACCELERATOR_TYPE:-"NVIDIA_TESLA_T4"}
ACCELERATOR_COUNT=${ACCELERATOR_COUNT:-1}
DISPLAY_NAME="pugmark-training-$(date +%Y%m%d-%H%M%S)"
CONTAINER_URI="us-central1-docker.pkg.dev/${PROJECT_ID}/pugmark/pugmark:latest"
REPLICA_COUNT=1
JOB_DIR="gs://${PROJECT_ID}-ml/pugmark/training/$(date +%Y%m%d-%H%M%S)"

echo "Submitting Vertex AI custom training job with the following configuration:"
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Machine Type: ${MACHINE_TYPE}"
echo "Accelerator Type: ${ACCELERATOR_TYPE}"
echo "Accelerator Count: ${ACCELERATOR_COUNT}"
echo "Display Name: ${DISPLAY_NAME}"
echo "Container URI: ${CONTAINER_URI}"
echo "Job Directory: ${JOB_DIR}"

# Create a Python script to submit the job
cat > submit_vertex_job.py << 'EOL'
import os
from google.cloud import aiplatform

# Configuration
project_id = os.environ["GCP_PROJECT_ID"]
region = "us-central1"
display_name = os.environ["DISPLAY_NAME"]
container_uri = os.environ["CONTAINER_URI"]
machine_type = os.environ["MACHINE_TYPE"]
accelerator_type = os.environ["ACCELERATOR_TYPE"]
accelerator_count = int(os.environ["ACCELERATOR_COUNT"])
replica_count = int(os.environ["REPLICA_COUNT"])
job_dir = os.environ["JOB_DIR"]

# Initialize Vertex AI SDK
aiplatform.init(project=project_id, location=region)

# Map accelerator type string to enum value
accelerator_map = {
    "NVIDIA_TESLA_T4": aiplatform.AcceleratorType.NVIDIA_TESLA_T4,
    "NVIDIA_TESLA_V100": aiplatform.AcceleratorType.NVIDIA_TESLA_V100,
    "NVIDIA_TESLA_P100": aiplatform.AcceleratorType.NVIDIA_TESLA_P100,
    "NVIDIA_TESLA_P4": aiplatform.AcceleratorType.NVIDIA_TESLA_P4,
}

# Create and run custom training job
job = aiplatform.CustomContainerTrainingJob(
    display_name=display_name,
    container_uri=container_uri,
)

# Start the training
model = job.run(
    args=["python3", "/app/src/training/train.py"],
    replica_count=replica_count,
    machine_type=machine_type,
    accelerator_type=accelerator_map[accelerator_type],
    accelerator_count=accelerator_count,
    boot_disk_type="pd-ssd",
    boot_disk_size_gb=100,
    base_output_dir=job_dir,
    service_account=f"github-service-account@{project_id}.iam.gserviceaccount.com",
    enable_web_access=True,  # Enable Vertex AI Workbench
    sync=False,  # Run asynchronously
)

print(f"Training job started: {job.resource_name}")
print(f"View job in the Cloud Console:")
print(f"https://console.cloud.google.com/vertex-ai/training/custom-jobs/{job.resource_name.split('/')[-1]}?project={project_id}")
EOL

# Export environment variables for the Python script
export GCP_PROJECT_ID=${PROJECT_ID}
export REGION=${REGION}
export DISPLAY_NAME=${DISPLAY_NAME}
export CONTAINER_URI=${CONTAINER_URI}
export MACHINE_TYPE=${MACHINE_TYPE}
export ACCELERATOR_TYPE=${ACCELERATOR_TYPE}
export ACCELERATOR_COUNT=${ACCELERATOR_COUNT}
export REPLICA_COUNT=${REPLICA_COUNT}
export JOB_DIR=${JOB_DIR}

# Run the Python script
python submit_vertex_job.py

echo "Job submission complete!" 