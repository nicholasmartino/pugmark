#!/usr/bin/env python3
"""
Execute a notebook on Vertex AI Workbench and wait for its completion.
This is a replacement for the Colab-based execution, providing more reliable execution
and monitoring capabilities through Google's managed Vertex AI platform.
"""
import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime

import nbformat
from google.cloud import aiplatform, storage


def setup_argparse():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a notebook on Vertex AI Workbench"
    )
    parser.add_argument("--notebook", required=True, help="Path to the input notebook")
    parser.add_argument(
        "--output", required=True, help="Path to save the executed notebook"
    )
    parser.add_argument(
        "--params", help="Parameters in the format PARAM1=VALUE1,PARAM2=VALUE2"
    )
    parser.add_argument(
        "--region", default="us-central1", help="Region for Vertex AI resources"
    )
    parser.add_argument(
        "--machine-type", default="n1-standard-4", help="Machine type for the instance"
    )
    parser.add_argument(
        "--accelerator-type",
        default="NVIDIA_TESLA_T4",
        help="Accelerator type (NVIDIA_TESLA_T4, NVIDIA_TESLA_V100, etc.)",
    )
    parser.add_argument(
        "--accelerator-count", type=int, default=1, help="Number of accelerators"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=14400,
        help="Timeout in seconds (default: 4 hours)",
    )
    parser.add_argument(
        "--bucket",
        help="GCS bucket to store temporary files (defaults to project-region bucket)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't attempt to open the browser (headless mode)",
    )

    return parser.parse_args()


def inject_parameters(notebook_path, params_str):
    """Inject parameters into the notebook."""
    if not params_str:
        return notebook_path

    # Parse parameters
    param_dict = {}
    for param in params_str.split(","):
        key, value = param.split("=", 1)
        param_dict[key] = value

    # Read the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    # Create a parameters cell at the beginning
    param_content = "# Parameters\n"
    for key, value in param_dict.items():
        if value.isdigit():
            param_content += f"{key} = {value}\n"
        else:
            param_content += f"{key} = '{value}'\n"

    param_cell = nbformat.v4.new_code_cell(param_content)
    param_cell["metadata"] = {"tags": ["parameters"]}

    # Insert at the beginning, after any existing parameter cells
    insertion_index = 0
    for i, cell in enumerate(notebook.cells):
        if "tags" in cell.metadata and "parameters" in cell.metadata["tags"]:
            insertion_index = i + 1

    notebook.cells.insert(insertion_index, param_cell)

    # Write to a temporary file
    temp_notebook_path = f"{notebook_path}.tmp"
    with open(temp_notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    return temp_notebook_path


def upload_to_gcs(local_path, bucket_name, blob_name):
    """Upload a file to Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"


def create_or_get_bucket(project_id, region):
    """Create or get a GCS bucket for temporary files."""
    bucket_name = f"{project_id}-{region}-vertex"
    storage_client = storage.Client()

    try:
        bucket = storage_client.get_bucket(bucket_name)
        print(f"Using existing bucket: {bucket_name}")
    except Exception:
        print(f"Creating new bucket: {bucket_name}")
        bucket = storage_client.create_bucket(bucket_name, location=region)

    return bucket_name


def execute_notebook(args):
    """Execute a notebook on Vertex AI and monitor until completion."""
    # Get project ID
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        # If not set in env var, try to get from credentials file
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path:
            with open(creds_path, "r") as f:
                creds = json.load(f)
                project_id = creds.get("project_id")

    if not project_id:
        raise ValueError(
            "Could not determine Google Cloud project ID. Please set GOOGLE_CLOUD_PROJECT environment variable."
        )

    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=args.region)

    # Get or create bucket
    bucket_name = args.bucket or create_or_get_bucket(project_id, args.region)

    # Create a unique ID for this execution
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    execution_id = f"train-{timestamp}-{run_id}"

    # Inject parameters if specified
    notebook_path = args.notebook
    if args.params:
        print(f"Injecting parameters: {args.params}")
        notebook_path = inject_parameters(args.notebook, args.params)

    # Upload notebook to GCS
    notebook_name = os.path.basename(notebook_path)
    gcs_path = upload_to_gcs(
        notebook_path, bucket_name, f"vertex-notebooks/{execution_id}/{notebook_name}"
    )
    print(f"Uploaded notebook to: {gcs_path}")

    # If we created a temporary notebook, clean it up
    if notebook_path != args.notebook:
        os.remove(notebook_path)

    # Add a verification cell to check for GPU
    gpu_verification = """
# GPU Verification
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPU Available:", len(gpus) > 0)
if len(gpus) > 0:
    print("GPU Devices:", [gpu.name for gpu in gpus])
    print("GPU Details:")
    !nvidia-smi
else:
    print("WARNING: No GPU detected! Training will be slow.")
"""

    # Create the CustomJobSpec with appropriate accelerator configuration
    worker_pool_spec = {
        "machine_spec": {
            "machine_type": args.machine_type,
            "accelerator_type": args.accelerator_type,
            "accelerator_count": args.accelerator_count,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "gcr.io/deeplearning-platform-release/tf-gpu.2-8:latest",
            "command": [
                "papermill",
                gcs_path,
                f"gs://{bucket_name}/vertex-notebooks/{execution_id}/output.ipynb",
                "-p",
                "EPOCHS",
                (
                    args.params.split("=")[1]
                    if args.params and "EPOCHS" in args.params
                    else "50"
                ),
            ],
        },
    }

    # Create a custom job to execute the notebook
    print(f"Creating Vertex AI custom job with GPU acceleration...")
    job = aiplatform.CustomJob.create(
        display_name=f"train-footprints-{execution_id}",
        worker_pool_specs=[worker_pool_spec],
        base_output_dir=f"gs://{bucket_name}/vertex-notebooks/{execution_id}/output",
    )

    print(f"Created Vertex AI job: {job.name}")
    print(
        f"Job details URL: https://console.cloud.google.com/vertex-ai/training/custom-jobs/{job.name.split('/')[-1]}?project={project_id}"
    )

    # Wait for job completion
    start_time = time.time()
    print(f"\n--- Waiting for notebook execution (timeout: {args.timeout}s) ---")

    try:
        # Poll for status
        status = "RUNNING"
        last_update_time = time.time()
        last_status = None

        while time.time() - start_time < args.timeout:
            job.reload()
            status = job.state

            # Only print when status changes
            if status != last_status:
                elapsed = time.time() - start_time
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Status: {status} (elapsed: {elapsed:.1f}s)"
                )
                last_status = status
                last_update_time = time.time()

            if status in ["SUCCEEDED", "COMPLETE", "COMPLETE_WITH_CONDITIONS"]:
                print(
                    f"\n✓ Job completed successfully after {time.time() - start_time:.1f} seconds"
                )
                break

            if status in ["FAILED", "CANCELLED"]:
                print(f"\n⚠️ Job failed with status: {status}")

                # Try to get error message
                try:
                    error_msg = job.error.message
                    print(f"Error details: {error_msg}")
                except:
                    print("No detailed error message available")

                return False

            # Check if we've had any updates recently
            if time.time() - last_update_time > 300:  # 5 minutes without updates
                print(
                    f"\n⚠️ No status updates in the last 5 minutes. Job may be stalled."
                )
                print(f"Current status is still: {status}")
                last_update_time = time.time()

            time.sleep(30)  # Check every 30 seconds

        if status not in ["SUCCEEDED", "COMPLETE", "COMPLETE_WITH_CONDITIONS"]:
            print(
                f"\n⚠️ Job did not complete within the timeout period ({args.timeout}s)."
            )
            return False

        # Download the results
        print("\nDownloading executed notebook...")
        output_blob = f"vertex-notebooks/{execution_id}/output.ipynb"
        try:
            download_from_gcs(bucket_name, output_blob, args.output)
            print(f"✓ Results saved to {args.output}")
        except Exception as e:
            print(f"⚠️ Error downloading results: {str(e)}")
            print("The job completed, but the results could not be downloaded.")
            print(
                f"You can access results in the GCS bucket: gs://{bucket_name}/{output_blob}"
            )
            return True  # Job still completed successfully

        return True

    except Exception as e:
        print(f"\n⚠️ Error monitoring job: {str(e)}")
        print("Check the Vertex AI console for details.")
        return False


def download_from_gcs(bucket_name, blob_name, local_path):
    """Download a file from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    blob.download_to_filename(local_path)
    return local_path


def main():
    """Main entry point."""
    args = setup_argparse()

    print("\n=== Vertex AI Notebook Executor ===")
    print(f"Input notebook: {args.notebook}")
    print(f"Output path: {args.output}")
    print(f"Machine type: {args.machine_type}")
    print(f"GPU: {args.accelerator_type} x{args.accelerator_count}")
    print(f"Region: {args.region}")
    print("=================================\n")

    success = execute_notebook(args)

    if success:
        print("\n✓ Notebook execution completed successfully")
        return 0
    else:
        print("\n⚠️ Notebook execution failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
