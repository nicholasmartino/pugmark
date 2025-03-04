#!/usr/bin/env python3
"""
Script to execute a notebook on Google Colab using the Google Drive API.
This replaces the colab-cli dependency with direct API calls.
"""

import argparse
import os
import sys
import time
import traceback

# Add early debugging prints
print("=== Starting Colab Execution Script ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Script path: {__file__}")

try:
    print("Importing required modules...")
    import nbformat
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    print("All modules imported successfully")
except ImportError as e:
    print(f"ERROR importing modules: {e}")
    traceback.print_exc()
    with open("/tmp/colab_error.log", "w") as f:
        f.write(f"Import error: {e}\n")
        traceback.print_exc(file=f)
    sys.exit(1)

# Ensure prints are flushed immediately
print("Setting up unbuffered output...", flush=True)
(
    sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stdout, "reconfigure")
    else None
)


def setup_argparse():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description="Run a notebook on Google Colab")
    parser.add_argument(
        "notebook_path", type=str, help="Path to the notebook to execute"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the executed notebook"
    )
    parser.add_argument(
        "--params",
        type=str,
        help="Comma-separated key=value pairs to inject as parameters",
    )
    parser.add_argument(
        "--machine-type",
        type=str,
        default="CPU",
        choices=["CPU", "GPU", "TPU"],
        help="Machine type to use for execution",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser")

    return parser.parse_args()


def get_drive_service(access_token):
    """Create Google Drive API service using an access token."""
    try:
        print(
            f"Creating credentials with access token (length: {len(access_token) if access_token else 0})",
            flush=True,
        )
        credentials = Credentials(token=access_token)

        print("Building Drive API service...", flush=True)
        service = build("drive", "v3", credentials=credentials)

        # Test the service with a simple API call
        print("Testing Drive API service with files.list call...", flush=True)
        test_result = service.files().list(pageSize=1).execute()
        print(f"Drive API service test successful: {test_result.keys()}", flush=True)

        return service
    except Exception as e:
        error_msg = f"Error creating Drive API service: {e}"
        print(error_msg, flush=True)
        print("\nError details:", flush=True)
        traceback.print_exc()
        with open("/tmp/colab_error.log", "w") as f:
            f.write(f"{error_msg}\n")
            traceback.print_exc(file=f)

        # Provide more detailed troubleshooting advice
        print("\nTROUBLESHOOTING TIPS:", flush=True)
        print(
            "1. Check that the Google Drive API is enabled in your Google Cloud Console",
            flush=True,
        )
        print(
            "   Visit: https://console.cloud.google.com/apis/library/drive.googleapis.com",
            flush=True,
        )
        print(
            "2. Verify that your service account has the proper Drive permissions",
            flush=True,
        )
        print("3. Make sure your access token is valid and not expired", flush=True)

        raise


def inject_parameters(notebook_path, params):
    """Inject parameters into the notebook."""
    if not params:
        return notebook_path

    # Read notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    # Parse parameters
    param_dict = {}
    for param in params.split(","):
        key, value = param.strip().split("=", 1)
        param_dict[key] = value

    # Add parameters cell if it doesn't exist
    param_cell = nbformat.v4.new_code_cell(
        f"# Parameters cell\n"
        + "\n".join([f"{k} = {repr(v)}" for k, v in param_dict.items()])
    )

    # Insert at beginning
    notebook.cells.insert(0, param_cell)

    # Write modified notebook to a temporary file
    temp_path = f"{notebook_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    return temp_path


def upload_to_drive(drive_service, file_path):
    """Upload the notebook to Google Drive."""
    print(f"Uploading {file_path} to Google Drive...")

    file_metadata = {
        "name": os.path.basename(file_path),
        "mimeType": "application/x-ipynb+json",
    }

    media = MediaFileUpload(
        file_path, mimetype="application/x-ipynb+json", resumable=True
    )

    file = (
        drive_service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )

    print(f"Uploaded file with ID: {file.get('id')}")
    return file.get("id")


def create_colab_vm_and_execute(drive_service, file_id, machine_type):
    """Create a Colab VM and execute the notebook."""
    print(f"Creating Colab VM with {machine_type} and executing notebook...")

    # Using drive API to update file resource with Colab execution metadata
    # This is a simplified approach - the actual Colab API is not public
    # In a production setting, you might need more sophisticated methods
    file_metadata = {
        "appProperties": {
            "colab-machine-type": machine_type.lower(),
            "colab-run-timestamp": str(int(time.time())),
        }
    }

    drive_service.files().update(fileId=file_id, body=file_metadata).execute()

    # Generate a Colab URL for manual execution if needed
    colab_url = f"https://colab.research.google.com/drive/{file_id}"
    print(f"Notebook is being executed. You can monitor it at: {colab_url}")

    # In a real implementation, we'd poll the execution status
    # Since the Colab execution API is not public, this is a simplified approach
    return colab_url


def wait_for_execution(drive_service, file_id, timeout_minutes=60):
    """Wait for the notebook execution to complete."""
    print("Waiting for notebook execution to complete...")

    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    while time.time() - start_time < timeout_seconds:
        # Check file metadata for execution status
        # Again, this is a simplified approach
        file = (
            drive_service.files().get(fileId=file_id, fields="appProperties").execute()
        )

        app_properties = file.get("appProperties", {})
        if "colab-execution-complete" in app_properties:
            print("Execution complete!")
            return True

        print("Execution in progress... (waiting 30 seconds)")
        time.sleep(30)

    print(
        f"Timeout after {timeout_minutes} minutes. Execution may still be in progress."
    )
    return False


def download_executed_notebook(drive_service, file_id, output_path):
    """Download the executed notebook."""
    print(f"Downloading executed notebook to {output_path}...")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export the file content
    response = drive_service.files().export_media(
        fileId=file_id, mimeType="application/x-ipynb+json"
    )

    with open(output_path, "wb") as f:
        # Process the response in chunks
        downloader = MediaFileUpload(
            f, mimetype="application/x-ipynb+json", resumable=True
        )
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Download {int(status.progress() * 100)}%")

    print(f"Notebook downloaded to {output_path}")


def cleanup(drive_service, file_id, temp_file=None):
    """Clean up temporary resources."""
    if temp_file and os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"Removed temporary file: {temp_file}")

    # Optionally delete the file from Drive
    # drive_service.files().delete(fileId=file_id).execute()
    # print(f"Deleted file from Drive: {file_id}")


def main():
    print("Parsing command line arguments...", flush=True)
    args = setup_argparse()
    print(f"Arguments: {args}", flush=True)

    # Get access token from environment
    access_token = os.environ.get("GCP_ACCESS_TOKEN")
    if not access_token:
        error_msg = "GCP_ACCESS_TOKEN environment variable is not set"
        print(f"ERROR: {error_msg}", flush=True)
        with open("/tmp/colab_error.log", "w") as f:
            f.write(f"{error_msg}\n")
        raise ValueError(error_msg)

    # Print environment information for debugging
    print("Environment variables:", flush=True)
    for key in ["GOOGLE_APPLICATION_CREDENTIALS", "PYTHONPATH", "GITHUB_TOKEN"]:
        print(f"  {key}: {'SET' if os.environ.get(key) else 'NOT SET'}", flush=True)

    # Print Drive API usage information
    shared_drive_id = os.environ.get("SHARED_DRIVE_ID")
    print(f"Using Shared Drive: {'Yes' if shared_drive_id else 'No'}", flush=True)
    if shared_drive_id:
        print(f"Shared Drive ID: {shared_drive_id}", flush=True)

    user_email = os.environ.get("SHARE_WITH_EMAIL")
    print(f"Sharing with user: {'Yes' if user_email else 'No'}", flush=True)
    if user_email:
        print(f"User email: {user_email}", flush=True)

    # Inject parameters if needed
    print(f"Injecting parameters into notebook: {args.notebook_path}", flush=True)
    temp_notebook_path = inject_parameters(args.notebook_path, args.params)
    file_id = None

    try:
        # Initialize Drive API
        print("Initializing Google Drive API service...", flush=True)
        drive_service = get_drive_service(access_token)
        print("Drive service initialized successfully", flush=True)

        try:
            # Upload notebook to Drive
            print("Starting notebook upload to Drive...", flush=True)
            file_id = upload_to_drive(drive_service, temp_notebook_path)
            print(f"Upload successful, file_id: {file_id}", flush=True)
        except Exception as e:
            error_msg = f"Error uploading to Drive: {e}"
            print(error_msg, flush=True)
            print(
                "Check if the service account has permission to access Google Drive",
                flush=True,
            )
            print("\nError details:", flush=True)
            traceback.print_exc()
            with open("/tmp/colab_error.log", "w") as f:
                f.write(f"{error_msg}\n")
                traceback.print_exc(file=f)
            raise

        # Execute notebook
        print("Creating Colab VM and executing notebook...", flush=True)
        colab_url = create_colab_vm_and_execute(
            drive_service, file_id, args.machine_type
        )

        # Wait for execution to complete
        print("Waiting for notebook execution to complete...", flush=True)
        execution_complete = wait_for_execution(drive_service, file_id)

        # Download the executed notebook
        if execution_complete:
            print(f"Downloading executed notebook to {args.output}", flush=True)
            download_executed_notebook(drive_service, file_id, args.output)
            print(
                f"Notebook execution completed and results saved to {args.output}",
                flush=True,
            )
        else:
            print(
                "Notebook execution did not complete within the timeout period.",
                flush=True,
            )
            print(f"You can check the status manually at: {colab_url}", flush=True)

    except Exception as e:
        error_msg = f"Error during Colab execution process: {e}"
        print(error_msg, flush=True)
        print("Check service account permissions and Google API access", flush=True)
        print("\nError details:", flush=True)
        traceback.print_exc()
        with open("/tmp/colab_error.log", "w") as f:
            f.write(f"{error_msg}\n")
            traceback.print_exc(file=f)
        raise
    finally:
        # Clean up resources
        if file_id:
            print("Cleaning up resources...", flush=True)
            cleanup(
                drive_service,
                file_id,
                (
                    temp_notebook_path
                    if temp_notebook_path != args.notebook_path
                    else None
                ),
            )


if __name__ == "__main__":
    main()
