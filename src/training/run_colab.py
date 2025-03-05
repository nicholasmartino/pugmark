#!/usr/bin/env python3
"""
Script to execute a notebook on Google Colab using the Google Drive API.
This replaces the colab-cli dependency with direct API calls.
"""

import argparse
import io
import os
import re
import sys
import time
import traceback

try:
    import nbformat

    # Import the notebook cells content
    try:
        from colab_cells import AUTORUN_CELL, LOG_STREAMING_CELL

        COLAB_CELLS_IMPORTED = True
    except ImportError:
        print(
            "WARNING: Could not import colab_cells.py - using default implementations"
        )
        COLAB_CELLS_IMPORTED = False
        LOG_STREAMING_CELL = """
# Default log streaming cell (colab_cells.py not found)
import sys
import time
import datetime

print("\\n" + "*" * 80)
print("* COLAB EXECUTION STREAMING INITIALIZED (DEFAULT VERSION) *".center(78))
print("*" * 80 + "\\n")
sys.stdout.flush()

print("[LOG] This is a basic version of the logging cell.")
print("[LOG] For full functionality, make sure colab_cells.py is available.")
sys.stdout.flush()
"""

        AUTORUN_CELL = """
# Default autorun cell (colab_cells.py not found)
import IPython

print("\\n" + "#" * 80)
print("# COLAB RUNTIME INFORMATION (DEFAULT VERSION) #".center(78))
print("#" * 80 + "\\n")
sys.stdout.flush()

# Show some basic system info
!nvidia-smi
!python --version
!hostname

print("\\n" + "*" * 80)
print("* EXECUTING ALL NOTEBOOK CELLS *".center(78))
print("*" * 80 + "\\n")
sys.stdout.flush()

# Run all cells
IPython.get_ipython().run_cell("from google.colab import runtime; runtime.connect()")
IPython.get_ipython().run_cell("%%capture\\n%run -i ../..")

print("\\n" + "*" * 80)
print("* NOTEBOOK EXECUTION COMPLETED *".center(78))
print("*" * 80 + "\\n")
sys.stdout.flush()
"""

    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload

    ALL_DEPENDENCIES_INSTALLED = True
except ImportError as e:
    print(f"ERROR: Missing dependency - {str(e)}")
    print("\nPlease install the required dependencies using:")
    print(
        "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib nbformat"
    )
    ALL_DEPENDENCIES_INSTALLED = False

# Add early debugging prints
print(f"Python version: {sys.version}", flush=True)
print(f"Script path: {os.path.abspath(__file__)}", flush=True)
print(f"Current directory: {os.getcwd()}", flush=True)


def setup_argparse():
    """Setup argument parsing with backward compatibility."""
    parser = argparse.ArgumentParser(
        description="Upload a notebook to Google Drive and execute it on Colab"
    )

    # Required positional argument for notebook path
    parser.add_argument("notebook_path", help="Path to the notebook file to execute")

    # Common parameters for both new and old interfaces
    parser.add_argument(
        "--machine-type",
        choices=["CPU", "GPU", "TPU"],
        default="GPU",
        help="Colab machine type (CPU, GPU, TPU)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout in minutes for notebook execution",
    )

    # New interface parameters
    parser.add_argument(
        "--checkpoint", help="Path to the checkpoint directory for training"
    )
    parser.add_argument("--epochs", type=int, help="Number of epochs for training")

    # Old interface parameters (for backward compatibility)
    parser.add_argument(
        "--params",
        help="Parameters to inject into the notebook in format PARAM1=VALUE1,PARAM2=VALUE2 (legacy)",
    )
    parser.add_argument("--output", help="Path to save executed notebook (legacy)")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser to check execution (legacy)",
    )

    return parser.parse_args()


def authenticate():
    """Authenticate with Google API using the provided credentials."""
    print("Authenticating with Google API...", flush=True)

    # Get access token from environment
    access_token = os.environ.get("GCP_ACCESS_TOKEN")
    if not access_token:
        error_msg = "GCP_ACCESS_TOKEN environment variable is not set"
        print(f"ERROR: {error_msg}", flush=True)
        raise ValueError(error_msg)

    # Create credentials using the access token
    credentials = Credentials(access_token)

    # Build the Drive API service
    drive_service = build("drive", "v3", credentials=credentials)

    # Verify the service works
    try:
        test_result = drive_service.files().list(pageSize=1).execute()
        print(f"Drive API service test successful", flush=True)
    except Exception as e:
        print(f"Error testing Drive API: {e}", flush=True)
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

    print("Successfully authenticated with Google API", flush=True)
    return credentials, drive_service


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
    print(f"Uploading {file_path} to Google Drive...", flush=True)

    # Get the shared drive ID from environment if provided
    shared_drive_id = os.environ.get("SHARED_DRIVE_ID")
    enable_public_access = (
        os.environ.get("ENABLE_PUBLIC_ACCESS", "false").lower() == "true"
    )

    if enable_public_access:
        print("Public access mode is enabled as a fallback", flush=True)

    file_metadata = {
        "name": os.path.basename(file_path),
        "mimeType": "application/x-ipynb+json",
    }

    # If using a shared drive, add it as parent and set supportsAllDrives flags
    params = {
        "fields": "id",
    }

    if shared_drive_id:
        file_metadata["parents"] = [shared_drive_id]
        params["supportsAllDrives"] = True
        print(f"Using Shared Drive with ID: {shared_drive_id}", flush=True)

    media = MediaFileUpload(
        file_path, mimetype="application/x-ipynb+json", resumable=True
    )

    print("Creating file in Drive...", flush=True)
    file = (
        drive_service.files()
        .create(body=file_metadata, media_body=media, **params)
        .execute()
    )

    file_id = file.get("id")
    print(f"Uploaded file with ID: {file_id}", flush=True)

    direct_sharing_successful = False

    # Share with a specific user if provided via environment variable
    user_email = os.environ.get("SHARE_WITH_EMAIL")
    if (
        user_email and not shared_drive_id
    ):  # No need to explicitly share if using a shared drive
        print(f"Attempting to share with user email: {user_email}", flush=True)

        # First verify if the email is valid
        try:
            # Check if @ exists in the email
            if "@" not in user_email:
                print(
                    f"Warning: Email address {user_email} appears to be invalid - missing @ symbol",
                    flush=True,
                )

            # Try different sharing approaches

            # Approach 1: Standard permission creation
            permission = {"type": "user", "role": "writer", "emailAddress": user_email}

            print("Creating permission with standard approach...", flush=True)
            result = (
                drive_service.permissions()
                .create(
                    fileId=file_id,
                    body=permission,
                    fields="id",
                    sendNotificationEmail=False,
                )
                .execute()
            )

            print(
                f"Standard sharing successful. Permission ID: {result.get('id')}",
                flush=True,
            )
            direct_sharing_successful = True

            # Generate Drive UI URL for direct access
            print("Generating access URLs...", flush=True)
            drive_ui_url = f"https://drive.google.com/file/d/{file_id}/view"
            colab_url = f"https://colab.research.google.com/drive/{file_id}"

            print(f"\n==== IMPORTANT: ACCESS LINKS ====", flush=True)
            print(f"Google Drive direct link: {drive_ui_url}", flush=True)
            print(f"Google Colab direct link: {colab_url}", flush=True)
            print(f"File has been shared with: {user_email}", flush=True)
            print(f"===============================\n", flush=True)

            # Verify sharing worked by checking permissions
            print("Verifying file permissions...", flush=True)
            permissions = (
                drive_service.permissions()
                .list(fileId=file_id, fields="permissions(id,emailAddress,role,type)")
                .execute()
            )

            print("Current permissions:", flush=True)
            for p in permissions.get("permissions", []):
                print(
                    f"  - {p.get('emailAddress', 'N/A')} ({p.get('role')})", flush=True
                )

        except Exception as e:
            print(f"Error sharing file with user: {str(e)}", flush=True)
            print("Detailed error info:", flush=True)
            traceback.print_exc()

    # If direct sharing failed or wasn't attempted, and public access is enabled
    if (not direct_sharing_successful or not user_email) and enable_public_access:
        print("\nAttempting to make file publicly accessible with link...", flush=True)

        try:
            # Make the file accessible to anyone with the link
            permission = {
                "type": "anyone",
                "role": "writer",
                "allowFileDiscovery": False,
            }

            drive_service.permissions().create(
                fileId=file_id, body=permission, fields="id"
            ).execute()

            print("File made accessible to anyone with the link", flush=True)
            print(f"\n==== IMPORTANT: PUBLIC ACCESS LINKS ====", flush=True)
            print(
                f"Google Drive link: https://drive.google.com/file/d/{file_id}/view",
                flush=True,
            )
            print(
                f"Google Colab link: https://colab.research.google.com/drive/{file_id}",
                flush=True,
            )
            print(
                f"Note: These links allow anyone with the link to access the file",
                flush=True,
            )
            print(f"===============================\n", flush=True)

        except Exception as e2:
            print(f"Public access sharing failed: {str(e2)}", flush=True)
            traceback.print_exc()
            print("\nTROUBLESHOOTING:", flush=True)
            print(
                "1. Make sure the Google Drive API is enabled: https://console.cloud.google.com/apis/library/drive.googleapis.com",
                flush=True,
            )
            print(
                "2. Check that your service account has the necessary permissions",
                flush=True,
            )
            print(
                "3. Verify that the USER_EMAIL secret is correctly set in your repository settings",
                flush=True,
            )

    return file_id


def create_colab_vm_and_execute(drive_service, file_id, machine_type):
    """Create a Colab VM and execute the notebook."""
    print(
        f"Creating Colab VM with {machine_type} and executing notebook...", flush=True
    )

    # Check if we're using a shared drive
    shared_drive_id = os.environ.get("SHARED_DRIVE_ID")
    params = {}
    if shared_drive_id:
        params["supportsAllDrives"] = True

    # Add a cell at the top that automatically executes all cells and streams logs
    try:
        # First, get the current notebook content
        print(
            "Getting notebook content to add autorun and logging cells...", flush=True
        )
        response = drive_service.files().get_media(fileId=file_id, **params).execute()

        if response:
            # Parse the notebook
            notebook_content = nbformat.reads(response.decode("utf-8"), as_version=4)

            # Create logging cell from imported code
            log_cell = nbformat.v4.new_code_cell(LOG_STREAMING_CELL)

            # Add autorun cell from imported code
            autorun_cell = nbformat.v4.new_code_cell(AUTORUN_CELL)

            # Insert at beginning (in reverse order so they appear in the right sequence)
            notebook_content.cells.insert(0, autorun_cell)
            notebook_content.cells.insert(0, log_cell)

            # Convert back to string
            updated_notebook = nbformat.writes(notebook_content)

            # Update the file in Drive - using MediaIoBaseUpload for BytesIO
            byte_content = io.BytesIO(updated_notebook.encode())
            media = MediaIoBaseUpload(
                byte_content, mimetype="application/x-ipynb+json", resumable=True
            )

            drive_service.files().update(
                fileId=file_id, media_body=media, **params
            ).execute()

            print("Added log streaming and autorun cells to notebook", flush=True)

    except Exception as e:
        print(f"Could not add streaming cells: {e}", flush=True)
        print(
            "The notebook will need to be executed manually and logs won't be streamed",
            flush=True,
        )
        traceback.print_exc()

    # Using drive API to update file resource with Colab execution metadata
    print("Setting Colab execution metadata...", flush=True)
    file_metadata = {
        "appProperties": {
            "colab-machine-type": machine_type.lower(),
            "colab-run-timestamp": str(int(time.time())),
        }
    }

    drive_service.files().update(fileId=file_id, body=file_metadata, **params).execute()

    # Generate UI access URLs
    colab_url = f"https://colab.research.google.com/drive/{file_id}"
    drive_url = f"https://drive.google.com/file/d/{file_id}/view"

    print("\n==== COLAB NOTEBOOK ACCESS ====", flush=True)
    print(f"Colab URL: {colab_url}", flush=True)
    print(f"Drive URL: {drive_url}", flush=True)
    print("==============================\n", flush=True)

    print("\n==== HOW TO CONNECT TO RUNTIME ====", flush=True)
    print("1. Click the Colab URL above to open the notebook", flush=True)
    print(
        "2. If not connected automatically, click the 'Connect' button in the top-right corner",
        flush=True,
    )
    print("3. If prompted, select 'Connect to a hosted runtime'", flush=True)
    print(
        "4. For GPU access, go to Runtime → Change runtime type → Hardware accelerator → GPU",
        flush=True,
    )
    print(
        "5. If the autorun cell didn't work, use Runtime → Run all (or press Ctrl+F9)",
        flush=True,
    )
    print("==============================\n", flush=True)

    # If using a shared drive, provide additional info
    if shared_drive_id:
        print(
            "\nNOTE: This notebook is in a Shared Drive. All team members with access to"
        )
        print(
            f"the Shared Drive (ID: {shared_drive_id}) can view and edit it.",
            flush=True,
        )

    return colab_url, "colab_execution_log.txt"  # Return log file name for streaming


def wait_for_execution(service, file_id, timeout_minutes, poll_interval_seconds=30):
    """
    Wait for notebook execution to complete by checking the notebook content directly.

    This function monitors the execution progress by polling the notebook and looking for
    specific markers that indicate completion or continued execution.

    Args:
        service: The Google Drive API service instance
        file_id: The ID of the notebook file to monitor
        timeout_minutes: Maximum time to wait (in minutes)
        poll_interval_seconds: How often to check the notebook status (in seconds)

    Returns:
        True if execution completed successfully, False if timeout occurred,
        "token_expired" if token expiration is detected
    """
    print("\nWaiting for Colab notebook execution to complete...")
    print(
        f"Will poll every {poll_interval_seconds} seconds for up to {timeout_minutes} minutes"
    )
    print(f"Notebook ID: {file_id}")

    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    # Keep track of last observed output markers
    last_cell_executed = None
    last_seen_output = None
    last_debug_output_time = time.time() - 120  # Start with printing debug output

    while True:
        # Check if we've exceeded timeout
        elapsed_seconds = time.time() - start_time
        if elapsed_seconds > timeout_seconds:
            print(f"\n⚠️ Timeout of {timeout_minutes} minutes exceeded!")
            print(f"The notebook execution may still be in progress.")
            print(
                f"You can check the notebook at: https://colab.research.google.com/drive/{file_id}"
            )
            return False

        try:
            # Get the current notebook content
            request = service.files().export(fileId=file_id, mimeType="text/html")
            notebook_html = request.execute().decode("utf-8")

            # Check for distinct markers that show up in notebook output cells
            completed_marker = "NOTEBOOK EXECUTION COMPLETED" in notebook_html
            training_finished_marker = "Training finished in" in notebook_html

            # Look for cell execution markers like: "▼▼▼ EXECUTING CELL 5/20 ▼▼▼"
            cell_execution_matches = re.findall(
                r"▼▼▼ EXECUTING CELL (\d+)/(\d+) ▼▼▼", notebook_html
            )
            cell_completion_matches = re.findall(
                r"▲▲▲ CELL (\d+)/(\d+) COMPLETED", notebook_html
            )

            # Count status messages to understand progress
            colab_status_count = notebook_html.count("[COLAB_STATUS]")
            colab_log_count = notebook_html.count("[COLAB_LOG]")

            # Extract the most recent output for debugging
            status_pattern = (
                r"\[COLAB_STATUS\].*?:(.+?)(?=\[COLAB_STATUS\]|\[COLAB_LOG\]|$)"
            )
            status_matches = re.findall(status_pattern, notebook_html, re.DOTALL)

            current_cell = None
            total_cells = None

            if cell_execution_matches:
                # Get the most recent cell being executed
                current_cell, total_cells = map(int, cell_execution_matches[-1])

            if cell_completion_matches:
                # Get the most recently completed cell
                last_completed, _ = map(int, cell_completion_matches[-1])
                if last_completed != last_cell_executed:
                    last_cell_executed = last_completed
                    print(
                        f"✓ Cell {last_completed}/{total_cells if total_cells else '?'} completed"
                    )

            # Print detailed progress information every 2 minutes or when significant changes happen
            current_time = time.time()
            if current_time - last_debug_output_time > 120 or (  # Every 2 minutes
                current_cell and last_cell_executed != current_cell
            ):

                print("\n📊 Notebook Execution Status:")
                print(f"├─ Elapsed time: {elapsed_seconds/60:.1f} minutes")

                if current_cell and total_cells:
                    progress_percent = (current_cell / total_cells) * 100
                    print(
                        f"├─ Progress: Cell {current_cell}/{total_cells} ({progress_percent:.1f}%)"
                    )
                else:
                    print(f"├─ Status messages detected: {colab_status_count}")
                    print(f"├─ Log messages detected: {colab_log_count}")

                # Show some recent output for debugging
                if status_matches:
                    recent_status = status_matches[-1].strip()
                    # Only print if it's different from last time
                    if recent_status != last_seen_output:
                        print(f"├─ Recent status: {recent_status}")
                        last_seen_output = recent_status

                # Extract any visible output
                output_pattern = r"<pre.*?>(.*?)</pre>"
                output_matches = re.findall(output_pattern, notebook_html, re.DOTALL)
                if output_matches:
                    # Get the last chunk of output, clean it up and truncate
                    recent_output = output_matches[-1].strip()
                    if len(recent_output) > 500:
                        recent_output = recent_output[-500:] + "..."
                    print(f"└─ Output preview: \n{recent_output}")

                last_debug_output_time = current_time

            # Check for authentication errors or token expiration
            auth_error_markers = [
                "not have permission to get",
                "Authentication Required",
                "authenticate to access",
                "Token has been expired",
                "credentials have expired",
            ]

            for marker in auth_error_markers:
                if marker in notebook_html:
                    print("\n🚨 Authentication error detected!")
                    print(
                        "The authentication token may have expired after 60+ minutes of execution."
                    )
                    print("This is a known limitation when running long training jobs.")
                    print("The notebook execution may have been interrupted.")
                    print(
                        f"Please check the notebook manually at: https://colab.research.google.com/drive/{file_id}"
                    )
                    return "token_expired"

            # Check for successful completion
            if completed_marker or training_finished_marker:
                print("\n✅ Notebook execution completed successfully!")
                if training_finished_marker:
                    # Try to extract training time
                    training_time_pattern = r"Training finished in (.+?)(?:\.|$)"
                    training_time_match = re.search(
                        training_time_pattern, notebook_html
                    )
                    if training_time_match:
                        print(f"Training completed in: {training_time_match.group(1)}")
                return True

            # Wait before checking again
            time.sleep(poll_interval_seconds)

        except HttpError as error:
            if error.resp.status == 401:
                print("\n🔑 Authentication token has expired!")
                print("This can happen when running long training jobs (60+ minutes).")
                print("The notebook execution may still be in progress.")
                print(
                    f"Please check the notebook manually at: https://colab.research.google.com/drive/{file_id}"
                )
                return "token_expired"
            else:
                print(f"Error checking notebook status: {error}")
                # For other errors, wait and retry
                time.sleep(poll_interval_seconds)

        except Exception as e:
            print(f"Unexpected error checking notebook status: {str(e)}")
            time.sleep(poll_interval_seconds)


def download_from_drive(drive_service, file_id, output_path):
    """Download the executed notebook from Drive."""
    print(f"Downloading file from Drive to {output_path}...", flush=True)

    # Check if we're using a shared drive
    shared_drive_id = os.environ.get("SHARED_DRIVE_ID")
    params = {}
    if shared_drive_id:
        params["supportsAllDrives"] = True

    try:
        # Get the file content
        request = drive_service.files().get_media(fileId=file_id, **params)
        content = request.execute()

        # Write content to output file
        with open(output_path, "wb") as f:
            f.write(content)

        print(f"Successfully downloaded notebook to {output_path}", flush=True)
        return True
    except Exception as e:
        print(f"Error downloading notebook: {e}", flush=True)
        traceback.print_exc()
        return False


def cleanup(drive_service, file_id, temp_file=None):
    """Clean up temporary resources."""
    if temp_file and os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"Removed temporary file: {temp_file}")

    # Optionally delete the file from Drive
    # drive_service.files().delete(fileId=file_id).execute()
    # print(f"Deleted file from Drive: {file_id}")


def main():
    """
    Main function to parse arguments and execute notebook
    """
    # Check if all dependencies are installed
    if not ALL_DEPENDENCIES_INSTALLED:
        sys.exit(1)

    args = setup_argparse()

    print(f"Notebook: {args.notebook_path}")

    # Process and map old-style parameters to new ones
    if args.params:
        print(f"Processing legacy parameters: {args.params}")
        # Extract EPOCHS from params string if present
        if "EPOCHS=" in args.params:
            epochs_match = re.search(r"EPOCHS=(\d+)", args.params)
            if epochs_match and not args.epochs:
                args.epochs = int(epochs_match.group(1))
                print(f"  - Extracted epochs: {args.epochs}")

        # You could extract other parameters here if needed

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Epochs: {args.epochs}")
    print(f"Timeout: {args.timeout} minutes")
    print(f"Machine type: {args.machine_type}")
    if args.output:
        print(f"Output path (legacy): {args.output}")

    # Set up environment variables for notebook execution
    if args.checkpoint:
        os.environ["CHECKPOINT_PATH"] = args.checkpoint
    if args.epochs:
        os.environ["EPOCHS"] = str(args.epochs)

    # Setup Google Drive API
    creds = authenticate()
    drive_service = build("drive", "v3", credentials=creds)

    # Upload the notebook to Google Drive
    file_id = upload_to_drive(drive_service, args.notebook_path)

    if not file_id:
        print("Failed to upload notebook to Google Drive.")
        sys.exit(1)

    # Create Colab VM and execute the notebook
    colab_url, log_file_name = create_colab_vm_and_execute(
        drive_service, file_id, args.machine_type
    )

    # Wait for execution to complete
    try:
        # Print the Colab URL
        print(
            f"Notebook is now executing on Colab: {colab_url}",
            flush=True,
        )

        # Wait for the notebook to finish executing and monitor its outputs
        status = wait_for_execution(
            drive_service,
            file_id,
            timeout_minutes=args.timeout,
        )

        # Handle downloading the executed notebook for backward compatibility
        if args.output and status != "token_expired":
            try:
                print(f"Downloading executed notebook to {args.output}...")
                download_from_drive(drive_service, file_id, args.output)
                print(f"Notebook saved to {args.output}")
            except Exception as e:
                print(f"Failed to download notebook: {str(e)}")

        # Handle different return values
        if status == "token_expired":
            print("Execution continued in Colab after authentication token expired.")
            print("Please check the notebook URL to view the complete results.")
            sys.exit(0)  # Exit with success as training continues in Colab
        elif status:
            print("Notebook execution completed successfully!")
            sys.exit(0)
        else:
            print("Notebook execution did not complete within the timeout period.")
            print("Check the notebook URL to view current progress.")
            sys.exit(1)

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
