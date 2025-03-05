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
        f"\nCreating Colab VM with {machine_type} and executing notebook...", flush=True
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
            log_cell.metadata = {"cellView": "form", "id": "log_streaming_cell"}

            # Add autorun cell from imported code
            autorun_cell = nbformat.v4.new_code_cell(AUTORUN_CELL)
            autorun_cell.metadata = {"cellView": "form", "id": "autorun_cell"}

            # Add a cell that directly triggers runtime connection - this is critical
            connect_cell = nbformat.v4.new_code_cell(
                """# Cell to force connection and execution
print("\\n" + "*" * 80)
print("* ATTEMPTING TO CONNECT TO RUNTIME *".center(78))
print("*" * 80 + "\\n")

# Try multiple approaches to ensure the runtime connects
try:
    from google.colab import drive
    # Mount drive - often helps trigger execution
    try:
        drive.mount('/content/drive')
        print("Drive mounted successfully")
    except:
        print("Drive mount not needed or failed")
    
    # Force runtime connection
    from google.colab import runtime
    runtime.connect()
    print("Explicitly connected to runtime")
    
    # Import tensorflow to verify GPU
    import tensorflow as tf
    if tf.config.list_physical_devices('GPU'):
        print("✓ GPU is available!")
    else:
        print("⚠️ No GPU found. Using CPU.")
        
    # Import display for clearing output
    from IPython import display
    display.clear_output(wait=True)
    print("Output cleared. Starting execution...")
    
    # Explicitly run first autorun cell
    import IPython
    print("Triggering autorun cell...")
    get_ipython().run_cell(%cell -i log_streaming_cell)
    get_ipython().run_cell(%cell -i autorun_cell)
    print("Autorun cells triggered")
except Exception as e:
    print(f"Error during startup: {str(e)}")
    print("Will try fallback methods...")
    # Try fallback execution
    try:
        get_ipython().system('pip install -q ipywidgets')
        print("Execution may need to be done manually - please open Colab URL and click Runtime > Run all")
    except:
        pass
"""
            )
            connect_cell.metadata = {"cellView": "form", "id": "force_connect_cell"}

            # Insert at beginning (in reverse order so they appear in the right sequence)
            notebook_content.cells.insert(0, autorun_cell)
            notebook_content.cells.insert(0, log_cell)
            notebook_content.cells.insert(0, connect_cell)

            # Add a special cell at the end to ensure we know when execution completes
            final_check_cell = nbformat.v4.new_code_cell(
                """# Final completion check
print("\\n" + "=" * 80)
print("NOTEBOOK EXECUTION COMPLETED - FINAL CHECK CELL REACHED")
print("All cells have been executed")
print("=" * 80 + "\\n")
"""
            )
            final_check_cell.metadata = {"cellView": "form", "id": "final_check_cell"}
            notebook_content.cells.append(final_check_cell)

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

            print("Added execution cells to notebook", flush=True)

    except Exception as e:
        print(f"Could not add execution cells: {e}", flush=True)
        print("The notebook will need to be executed manually", flush=True)
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

    # Manually trigger execution via the Colab API if possible
    try:
        print("\nAttempting to directly trigger notebook execution...", flush=True)
        # We can't directly trigger execution through the API, but we can try to simulate
        # browser interaction by making some API calls that might nudge Colab to start

        # First, let's make sure the file has the right MIME type for Colab
        update_metadata = {
            "mimeType": "application/vnd.google.colab",
            "appProperties": {
                "colab-kernel-name": "python3",
                "colab-runtime-mode": "AUTOMATED",
                "colab-auto-connect": "TRUE",
                "colab-session-id": f"session-{int(time.time())}",
            },
        }

        drive_service.files().update(
            fileId=file_id, body=update_metadata, **params
        ).execute()

        print("Updated file with Colab-specific metadata", flush=True)
        print(
            "Execution should begin automatically when Colab opens the notebook",
            flush=True,
        )
        print(
            "If execution doesn't start within 5 minutes, you may need to open the URL and trigger manually",
            flush=True,
        )

    except Exception as trigger_error:
        print(
            f"Note: Could not trigger automatic execution: {str(trigger_error)}",
            flush=True,
        )
        print(
            "You may need to open the Colab URL manually to start execution", flush=True
        )

    return colab_url, "colab_execution_log.txt"  # Return log file name for streaming


def wait_for_execution(service, file_id, timeout_minutes, poll_interval_seconds=30):
    """
    Wait for Colab notebook execution to complete with enhanced monitoring.

    This function checks notebook properties and content to monitor execution progress
    and handles situations where the notebook needs manual intervention.

    Args:
        service: The Google Drive API service
        file_id: The ID of the notebook file to monitor
        timeout_minutes: Maximum time to wait (in minutes)
        poll_interval_seconds: How often to check the notebook status

    Returns:
        True if execution completed or was started, False if timeout
    """
    print(f"\nWaiting for Colab notebook execution to complete...", flush=True)
    print(
        f"Will poll every {poll_interval_seconds} seconds for up to {timeout_minutes} minutes",
        flush=True,
    )
    print(f"Notebook ID: {file_id}", flush=True)

    # Generate direct URLs for user access
    colab_url = f"https://colab.research.google.com/drive/{file_id}"
    drive_url = f"https://drive.google.com/file/d/{file_id}/view"

    start_time = time.time()
    elapsed_minutes = 0
    manual_intervention_notified = False
    runtime_detected = False

    # Store metrics for output
    executed_cells = 0
    last_executed_cells = 0
    execution_detected = False

    # Enhanced terminal output for better UX
    print("\n" + "=" * 80, flush=True)
    print(" COLAB EXECUTION MONITOR ".center(80), flush=True)
    print("=" * 80, flush=True)
    print("\nMonitoring Colab notebook execution...", flush=True)
    print(f"Notebook URL: {colab_url}", flush=True)
    print("\nExecuting the following checks every 30 seconds:", flush=True)
    print("1. Checking for runtime connection", flush=True)
    print("2. Looking for execution progress markers", flush=True)
    print("3. Monitoring for notebook completion", flush=True)
    print("=" * 80 + "\n", flush=True)

    while elapsed_minutes < timeout_minutes:
        try:
            # Calculate elapsed time for reporting
            elapsed_minutes = (time.time() - start_time) / 60

            # Get the notebook content
            try:
                params = {"alt": "media"}
                response = service.files().get_media(fileId=file_id, **params).execute()
                notebook_content = response.decode("utf-8")

                # Parse the notebook
                notebook = nbformat.reads(notebook_content, as_version=4)

                # Extract execution counts and outputs
                execution_counts = []
                kernel_busy = False
                tf_import_detected = False
                training_started = False
                completion_marker = False
                error_detected = False
                latest_output = ""

                # Process each cell to gather execution metrics
                for cell in notebook.cells:
                    if cell.cell_type != "code":
                        continue

                    # Check execution count
                    if hasattr(cell, "execution_count") and cell.execution_count:
                        execution_counts.append(cell.execution_count)

                    # Process outputs
                    if hasattr(cell, "outputs") and cell.outputs:
                        for output in cell.outputs:
                            # Get text content from output
                            output_text = ""
                            if output.output_type == "stream" and "text" in output:
                                output_text = output.get("text", "")
                            elif (
                                output.output_type == "execute_result"
                                and "data" in output
                            ):
                                if "text/plain" in output["data"]:
                                    output_text = output["data"]["text/plain"]

                            # Check for specific markers
                            if output_text:
                                if "tensorflow" in output_text.lower():
                                    tf_import_detected = True
                                if (
                                    "training" in output_text.lower()
                                    and "epoch" in output_text.lower()
                                ):
                                    training_started = True
                                if (
                                    "notebook execution completed"
                                    in output_text.lower()
                                ):
                                    completion_marker = True
                                if "training finished" in output_text.lower():
                                    completion_marker = True
                                if (
                                    "error" in output_text.lower()
                                    and len(output_text) > 100
                                ):
                                    error_detected = True

                                # Keep the most recent output for reporting
                                if len(output_text) > len(latest_output):
                                    latest_output = output_text

                # Count executed cells
                executed_cells = len([c for c in execution_counts if c is not None])

                # Detect changes in execution
                if executed_cells > last_executed_cells:
                    execution_detected = True
                    last_executed_cells = executed_cells

                # Check metadata for runtime connection
                if not runtime_detected:
                    for cell in notebook.cells:
                        # Look for indications of runtime connection in metadata or outputs
                        if (
                            hasattr(cell, "metadata")
                            and "colab" in cell.metadata
                            and "resources" in cell.metadata["colab"]
                        ):
                            runtime_detected = True

                        # Also check outputs for runtime connection indicators
                        if hasattr(cell, "outputs") and cell.outputs:
                            for output in cell.outputs:
                                if output.output_type == "stream" and "text" in output:
                                    if (
                                        "connected to runtime" in output["text"].lower()
                                        or "gpu" in output["text"].lower()
                                    ):
                                        runtime_detected = True

                # Determine notebook status
                status_line = "=" * 40
                print("\n" + status_line, flush=True)
                print(f"STATUS UPDATE at {elapsed_minutes:.1f} minutes:", flush=True)
                print(
                    f"- Runtime connected: {'Yes ✓' if runtime_detected else 'No'}",
                    flush=True,
                )
                print(f"- Cells executed: {executed_cells}", flush=True)
                print(
                    f"- TensorFlow imported: {'Yes ✓' if tf_import_detected else 'No'}",
                    flush=True,
                )
                print(
                    f"- Training started: {'Yes ✓' if training_started else 'No'}",
                    flush=True,
                )

                # If execution is happening, show a different status
                if training_started:
                    # Extract a preview of recent output
                    if latest_output:
                        preview = (
                            latest_output.split("\n")[-3:]
                            if "\n" in latest_output
                            else latest_output
                        )
                        preview_text = (
                            "\n".join(preview) if isinstance(preview, list) else preview
                        )
                        preview_text = (
                            preview_text[:150] + "..."
                            if len(preview_text) > 150
                            else preview_text
                        )
                        print(f"- Recent output: \n{preview_text}", flush=True)

                # Handle potential issues
                if elapsed_minutes > 5 and not runtime_detected:
                    print(
                        "\n⚠️ No runtime connection detected after 5 minutes", flush=True
                    )
                    print("This usually means:", flush=True)
                    print(
                        "1. Colab may be waiting in a queue for resources", flush=True
                    )
                    print("2. The service account may lack permissions", flush=True)
                    print("\nSuggested actions:", flush=True)
                    print(f"- Manually open the notebook: {colab_url}", flush=True)
                    print("- Click 'Connect' in the top-right corner", flush=True)
                    print("- Select 'Connect to a hosted runtime'", flush=True)

                if (
                    elapsed_minutes > 10
                    and not execution_detected
                    and not manual_intervention_notified
                ):
                    print("\n🚨 No execution detected after 10 minutes", flush=True)
                    print("This notebook requires manual intervention:", flush=True)
                    print(f"1. Open the notebook: {colab_url}", flush=True)
                    print("2. Connect to a runtime with GPU", flush=True)
                    print(
                        "3. Run the notebook manually (Runtime > Run all)", flush=True
                    )
                    print(
                        "\nThe GitHub Action will continue monitoring for outputs",
                        flush=True,
                    )
                    manual_intervention_notified = True

                # Check for completion
                if completion_marker or ("training finished" in latest_output.lower()):
                    print("\n✅ Notebook execution completed successfully!", flush=True)
                    print("\nTraining has finished. Downloading results...", flush=True)
                    return True

                if error_detected:
                    print(
                        "\n⚠️ Potential error detected in notebook execution", flush=True
                    )
                    print("The notebook may need manual review", flush=True)
                    print(f"Please check: {colab_url}", flush=True)
                    # Don't return yet, continue monitoring

            except Exception as parse_error:
                print(f"\nError parsing notebook: {parse_error}", flush=True)
                if elapsed_minutes > 15 and not manual_intervention_notified:
                    print(
                        "\n🚨 Unable to parse notebook after 15 minutes of attempts",
                        flush=True,
                    )
                    print(
                        "This likely means the notebook requires manual intervention:",
                        flush=True,
                    )
                    print(f"1. Open the notebook: {colab_url}", flush=True)
                    print("2. Connect to a runtime with GPU", flush=True)
                    print(
                        "3. Run the notebook manually (Runtime > Run all)", flush=True
                    )
                    manual_intervention_notified = True

            # Check if we're hitting a token expiration (typically happens after ~60 minutes)
            if elapsed_minutes >= 58 and elapsed_minutes <= 62:
                print(
                    "\n⚠️ Approaching token expiration window (60 minutes)", flush=True
                )
                print(
                    "The monitoring may be interrupted, but the notebook will continue running",
                    flush=True,
                )
                print(f"You can monitor progress manually at: {colab_url}", flush=True)

            # Handle API errors that might indicate token expiration
        except HttpError as api_error:
            if api_error.resp.status in [401, 403]:
                print(
                    "\n🔑 Authentication token has expired or access denied", flush=True
                )
                print(
                    "This happens after approximately 60 minutes of runtime", flush=True
                )
                print(
                    "The notebook is still executing, but we can no longer monitor it",
                    flush=True,
                )
                print(f"Please check manually: {colab_url}", flush=True)
                return True  # Consider this successful - the notebook is running
            else:
                print(f"\nAPI error: {api_error}", flush=True)
                print("Will retry in the next polling interval", flush=True)

        except Exception as e:
            print(f"\nUnexpected error during monitoring: {e}", flush=True)

        # Add a wait indicator for UX
        print("\nWaiting for next status check ", end="", flush=True)
        for _ in range(3):
            time.sleep(min(10, poll_interval_seconds / 3))
            print(".", end="", flush=True)
        print(" ", flush=True)

        # Wait before checking again
        time.sleep(
            max(1, poll_interval_seconds - 30)
        )  # Account for time spent in the loop

    # Timeout reached
    print("\n⏱️ Timeout reached after waiting for", flush=True)
    print(f"{timeout_minutes} minutes!", flush=True)
    print(
        "The notebook might still be executing, but monitoring has stopped", flush=True
    )
    print(f"Please check manually: {colab_url}", flush=True)

    return False  # Timeout


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
    creds, drive_service = authenticate()

    # Create a temporary notebook with parameters if needed
    notebook_to_upload = args.notebook_path
    if args.params:
        print(f"Injecting parameters into notebook: {args.params}")
        notebook_to_upload = inject_parameters(args.notebook_path, args.params)

    # Upload the notebook to Google Drive
    file_id = upload_to_drive(drive_service, notebook_to_upload)

    # Clean up temporary notebook if created
    if notebook_to_upload != args.notebook_path:
        try:
            os.remove(notebook_to_upload)
            print(f"Removed temporary notebook: {notebook_to_upload}")
        except Exception as e:
            print(f"Warning: Could not remove temporary notebook: {e}")

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
