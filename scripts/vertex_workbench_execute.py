#!/usr/bin/env python3
"""
Execute training code on a Vertex AI Workbench instance.
This script can create a new Workbench instance or use an existing one,
then execute training code via SSH.
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time

from google.cloud import aiplatform
from google.cloud.notebooks_v1 import NotebookServiceClient
from google.cloud.notebooks_v1.types import AcceleratorConfig, Instance

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("workbench-executor")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Execute training on Vertex AI Workbench"
    )
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument(
        "--region",
        default="us-central1",
        help="Google Cloud region for the notebook instance",
    )
    parser.add_argument(
        "--zone",
        default="us-central1-a",
        help="Google Cloud zone for the notebook instance",
    )
    parser.add_argument(
        "--instance-name",
        required=True,
        help="Name of the Vertex AI Workbench instance",
    )
    parser.add_argument(
        "--create-instance",
        default="true",
        choices=["true", "false"],
        help="Create a new instance if it doesn't exist",
    )
    parser.add_argument(
        "--machine-type",
        default="n1-standard-8",
        help="Machine type for the notebook instance",
    )
    parser.add_argument(
        "--accelerator-type",
        default="NVIDIA_TESLA_T4",
        choices=[
            "NVIDIA_TESLA_T4",
            "NVIDIA_TESLA_V100",
            "NVIDIA_TESLA_P100",
            "NVIDIA_TESLA_P4",
        ],
        help="GPU accelerator type",
    )
    parser.add_argument(
        "--accelerator-count",
        type=int,
        default=1,
        choices=[1, 2, 4, 8],
        help="Number of GPUs",
    )
    parser.add_argument(
        "--delete-after-training",
        default="false",
        choices=["true", "false"],
        help="Delete the instance after training completes",
    )
    parser.add_argument(
        "--output-bucket",
        default="${PROJECT_ID}-ml",
        help="GCS bucket to store training outputs",
    )
    parser.add_argument(
        "--output-prefix",
        default="pugmark/training",
        help="Prefix within the bucket for training outputs",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument(
        "--repo-url",
        default="",
        help="Repository URL to clone. If empty, uses the current GitHub repository.",
    )

    return parser.parse_args()


def get_or_create_instance(
    project_id,
    region,
    zone,
    instance_name,
    machine_type,
    accelerator_type,
    accelerator_count,
    create_instance=False,
):
    """Get or create a Vertex AI Workbench instance with GPU."""
    client = NotebookServiceClient()
    parent = f"projects/{project_id}/locations/{region}"
    instance_path = f"{parent}/instances/{instance_name}"

    # Check if instance exists
    try:
        instance = client.get_instance(name=instance_path)
        logger.info(f"Using existing instance: {instance_name}")
        instance_created = False
        return instance_path, instance, instance_created
    except Exception as e:
        logger.info(f"Instance {instance_name} not found: {e}")
        if not create_instance:
            raise ValueError(
                f"Instance {instance_name} does not exist and create_instance is False"
            )

    logger.info(f"Creating new Vertex AI Workbench instance: {instance_name}")

    # Map accelerator type to enum
    accelerator_map = {
        "NVIDIA_TESLA_T4": AcceleratorConfig.AcceleratorType.NVIDIA_TESLA_T4,
        "NVIDIA_TESLA_V100": AcceleratorConfig.AcceleratorType.NVIDIA_TESLA_V100,
        "NVIDIA_TESLA_P100": AcceleratorConfig.AcceleratorType.NVIDIA_TESLA_P100,
        "NVIDIA_TESLA_P4": AcceleratorConfig.AcceleratorType.NVIDIA_TESLA_P4,
    }

    # Create instance configuration
    instance = Instance(
        name=instance_path,
        machine_type=machine_type,
        accelerator_config=AcceleratorConfig(
            type_=accelerator_map.get(accelerator_type), core_count=accelerator_count
        ),
        vm_image=Instance.VmImage(
            project="deeplearning-platform-release",
            image_name="tf-ent-2-12-cu113-notebooks-v20230627-debian-11",
        ),
        boot_disk_size_gb=100,
        boot_disk_type="PD_SSD",
        install_gpu_driver=True,
        metadata={
            "install-nvidia-driver": "True",
            "proxy-mode": "service_account",
            "tf-enable-gpu": "True",
        },
    )

    # Create the instance
    operation = client.create_instance(
        parent=parent, instance_id=instance_name, instance=instance
    )
    logger.info("Creating instance... This may take several minutes.")
    instance = operation.result()
    logger.info(f"Instance created: {instance.name}")

    # Wait for instance to be ready
    logger.info("Waiting for instance to initialize...")
    time.sleep(120)  # Wait 2 minutes for initialization

    instance_created = True
    return instance_path, instance, instance_created


def create_startup_script(repo_url, output_path, epochs, batch_size):
    """Create a startup script to run the training job."""
    script = f"""#!/bin/bash
set -e

# Clone repository if specified
if [ -n "{repo_url}" ]; then
  echo "Cloning repository {repo_url}..."
  rm -rf /tmp/pugmark
  git clone {repo_url} /tmp/pugmark
  cd /tmp/pugmark
  pip install -r requirements.txt
else
  echo "Using existing repository..."
fi

# Set environment variables
export OUTPUT_DIR="{output_path}"
export AIP_MODE="training"
export TF_FORCE_GPU_ALLOW_GROWTH="true"
export EPOCHS="{epochs}"
export BATCH_SIZE="{batch_size}"

# Check GPU
echo "Checking GPU availability..."
nvidia-smi || echo "No GPU found"
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU devices:', tf.config.list_physical_devices('GPU'))"

# Create output directory
mkdir -p {output_path}

# Run training
echo "Starting training..."
cd /tmp/pugmark
python -c "from src.training.Trainer import train; train()"

echo "Training complete!"
echo "TRAINING_COMPLETED" > /tmp/training_completed.flag
"""
    return script


def execute_on_instance(project_id, zone, instance_name, startup_script):
    """Execute a command on the Workbench instance."""
    # Write startup script to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmp:
        tmp.write(startup_script)
        tmp_path = tmp.name

    # Use gcloud to execute the script on the instance
    cmd = [
        "gcloud",
        "compute",
        "ssh",
        instance_name,
        "--project",
        project_id,
        "--zone",
        zone,
        "--command",
        f"bash -s < {tmp_path}",
    ]

    try:
        logger.info(f"Executing command on instance {instance_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Command output: {result.stdout}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {e.stderr}")
        raise
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)


def delete_instance(project_id, region, instance_name):
    """Delete a Vertex AI Workbench instance."""
    client = NotebookServiceClient()
    instance_path = (
        f"projects/{project_id}/locations/{region}/instances/{instance_name}"
    )

    try:
        logger.info(f"Deleting instance {instance_name}...")
        operation = client.delete_instance(name=instance_path)
        operation.result()  # Wait for the operation to complete
        logger.info(f"Instance {instance_name} deleted successfully.")
        return True
    except Exception as e:
        logger.error(f"Error deleting instance: {e}")
        return False


def main():
    """Main function."""
    args = parse_args()

    # Initialize Vertex AI SDK
    aiplatform.init(project=args.project_id, location=args.region)

    # Format output path
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_bucket = args.output_bucket.replace("${PROJECT_ID}", args.project_id)
    output_path = f"gs://{output_bucket}/{args.output_prefix}/{timestamp}"

    # Get repository URL if not specified
    repo_url = args.repo_url
    if not repo_url and "GITHUB_REPOSITORY" in os.environ:
        github_server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        repo_url = f"{github_server}/{os.environ['GITHUB_REPOSITORY']}"

    create_instance_flag = args.create_instance.lower() == "true"
    delete_after_training_flag = args.delete_after_training.lower() == "true"

    instance_created = False
    try:
        # Get or create Workbench instance
        instance_path, instance, instance_created = get_or_create_instance(
            args.project_id,
            args.region,
            args.zone,
            args.instance_name,
            args.machine_type,
            args.accelerator_type,
            args.accelerator_count,
            create_instance_flag,
        )

        # Get instance zone if it wasn't created by us
        if not instance_created:
            try:
                # Get zone from instance metadata
                zone = instance.name.split("/")[-3]
            except (AttributeError, IndexError):
                # Fallback to the provided zone
                zone = args.zone
        else:
            zone = args.zone

        # Wait for instance to be fully ready
        logger.info(f"Waiting for instance {args.instance_name} to be ready...")
        time.sleep(60)  # Wait for SSH to be available

        # Create startup script
        startup_script = create_startup_script(
            repo_url, output_path, args.epochs, args.batch_size
        )

        # Execute training on the instance
        execute_on_instance(args.project_id, zone, args.instance_name, startup_script)

        logger.info(f"Training job submitted to instance {args.instance_name}")
        logger.info(f"Output will be stored at: {output_path}")

        # Delete instance if requested and we created it
        if delete_after_training_flag and instance_created:
            logger.info("Training completed. Deleting instance as requested...")
            delete_instance(args.project_id, args.region, args.instance_name)

    except Exception as e:
        logger.error(f"Error executing training on Workbench: {e}", exc_info=True)

        # Cleanup if requested and we created the instance
        if instance_created and delete_after_training_flag:
            logger.info("Cleaning up: Deleting created instance due to error...")
            delete_instance(args.project_id, args.region, args.instance_name)

        sys.exit(1)


if __name__ == "__main__":
    main()
