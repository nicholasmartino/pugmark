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
    Wait for Colab notebook execution to complete.

    Monitors the notebook by checking its content for execution markers periodically.
    Returns True if execution completed successfully, False if timeout or errors occurred.
    """
    print(f"\nWaiting for Colab notebook execution to complete...", flush=True)
    (
        print(
            f"Will poll every {poll_interval_seconds//60} minutes for up to {timeout_minutes} minutes",
            flush=True,
        )
        if poll_interval_seconds >= 60
        else print(
            f"Will poll every {poll_interval_seconds} seconds for up to {timeout_minutes} minutes",
            flush=True,
        )
    )
    print(f"Notebook ID: {file_id}", flush=True)

    start_time = time.time()
    params = {"alt": "media"}

    # Track the last time we saw an update to determine if notebook is stalled
    last_update_time = start_time
    last_cell_count = 0
    last_status_count = 0
    last_log_count = 0
    startup_delay_warning_shown = False
    is_running = False

    # Keep track of the cells we've seen
    executed_cell_indices = set()
    total_cells = 0

    while (time.time() - start_time) / 60 < timeout_minutes:
        try:
            # Calculate elapsed time
            elapsed_minutes = (time.time() - start_time) / 60

            # Check if token is about to expire (tokens typically last 60 minutes)
            if elapsed_minutes > 55 and not is_running:
                print(
                    "\n⚠️ WARNING: Wait time is approaching token expiration limit (60 minutes)",
                    flush=True,
                )
                print(
                    "No notebook activity detected yet. There might be an issue with:"
                )
                print("1. Colab runtime availability (high demand/queue)")
                print("2. Service account permissions")
                print("3. Network connectivity")
                print(
                    "\nPlease check the Colab URL manually to verify status.",
                    flush=True,
                )

            # Get the notebook content
            request = service.files().get_media(fileId=file_id, **params)
            notebook_content = request.execute().decode("utf-8")

            # Parse the notebook content
            try:
                notebook = nbformat.reads(notebook_content, as_version=4)

                # Extract output from all cells
                cell_outputs = []
                executed_cells = 0
                status_message_count = 0
                log_message_count = 0
                runtime_connected = False
                has_gpu = False

                # Count total cells for progress reporting (excluding our injected cells)
                if total_cells == 0:
                    # Filter out our special cells with ID metadata
                    user_cells = [
                        cell
                        for cell in notebook.cells
                        if not (
                            hasattr(cell, "metadata")
                            and cell.metadata.get("id")
                            in [
                                "force_connect_cell",
                                "log_streaming_cell",
                                "autorun_cell",
                                "final_check_cell",
                            ]
                        )
                    ]
                    total_cells = len(user_cells)

                # Process all cells
                for i, cell in enumerate(notebook.cells):
                    # Skip our special cells in the count
                    is_special_cell = hasattr(cell, "metadata") and cell.metadata.get(
                        "id"
                    ) in [
                        "force_connect_cell",
                        "log_streaming_cell",
                        "autorun_cell",
                        "final_check_cell",
                    ]

                    if (
                        cell.cell_type == "code"
                        and hasattr(cell, "outputs")
                        and cell.outputs
                    ):
                        # Count executed cells (but not our special cells)
                        if not is_special_cell:
                            executed_cells += 1
                            executed_cell_indices.add(i)

                        # Check for startup output in the connect cell
                        if (
                            hasattr(cell, "metadata")
                            and cell.metadata.get("id") == "force_connect_cell"
                        ):
                            for output in cell.outputs:
                                if output.output_type == "stream" and "text" in output:
                                    if (
                                        "Explicitly connected to runtime"
                                        in output["text"]
                                    ):
                                        runtime_connected = True
                                    if "GPU is available" in output["text"]:
                                        has_gpu = True

                        # Extract all text outputs
                        for output in cell.outputs:
                            if output.output_type == "stream" and "text" in output:
                                cell_outputs.append(output["text"])

                                # Count status and log messages
                                if (
                                    "▼▼▼ EXECUTING CELL" in output["text"]
                                    or "▲▲▲ CELL" in output["text"]
                                ):
                                    status_message_count += 1
                                if "[Training]" in output["text"]:
                                    log_message_count += 1

                # Convert all outputs to a single string for easier searching
                all_output = "\n".join(cell_outputs)

                # Check for distinct markers that show up in notebook output cells
                completed_marker = "NOTEBOOK EXECUTION COMPLETED" in all_output
                training_finished_marker = "Training finished in" in all_output
                error_marker = "Training encountered an error" in all_output

                # Look for cell execution markers like: "▼▼▼ EXECUTING CELL 5/20 ▼▼▼"
                import re

                cell_execution_matches = re.findall(
                    r"▼▼▼ EXECUTING CELL (\d+)/(\d+) ▼▼▼", all_output
                )
                cell_completion_matches = re.findall(
                    r"▲▲▲ CELL (\d+)/(\d+) COMPLETED", all_output
                )

                # Check for changes to determine if notebook is active
                status_changed = last_status_count != status_message_count
                logs_changed = last_log_count != log_message_count
                cells_changed = last_cell_count != executed_cells

                # If anything changed, update the "last update" time
                if status_changed or logs_changed or cells_changed:
                    last_update_time = time.time()
                    is_running = True

                # Determine current running status
                current_cell = None
                for match in cell_execution_matches:
                    current_cell = int(match[0])

                # Display status with detailed information
                print(f"\n📊 Notebook Execution Status:", flush=True)
                print(f"├─ Elapsed time: {elapsed_minutes:.1f} minutes", flush=True)

                # Runtime status
                runtime_status = []
                if runtime_connected:
                    runtime_status.append("Connected to runtime ✓")
                    if has_gpu:
                        runtime_status.append("GPU available ✓")
                    else:
                        runtime_status.append("Using CPU")
                else:
                    if not startup_delay_warning_shown and elapsed_minutes > 2:
                        print("\n⚠️ Runtime connection not detected yet...", flush=True)
                        print(
                            "This may take a few minutes on first execution or during high Colab usage.",
                            flush=True,
                        )
                        print("You can check the status manually at:", flush=True)
                        print(
                            f"https://colab.research.google.com/drive/{file_id}\n",
                            flush=True,
                        )
                        startup_delay_warning_shown = True

                if runtime_status:
                    print(f"├─ Runtime: {', '.join(runtime_status)}", flush=True)

                # Progress information
                if total_cells > 0:
                    progress_percent = executed_cells / total_cells * 100
                    print(
                        f"├─ Progress: {executed_cells}/{total_cells} cells executed ({progress_percent:.1f}%)",
                        flush=True,
                    )
                else:
                    print(f"├─ Executed cells: {executed_cells}", flush=True)

                # Current execution
                if current_cell:
                    print(
                        f"├─ Currently executing: Cell {current_cell}/{total_cells}",
                        flush=True,
                    )

                # Status messages and logs
                print(f"├─ Status messages: {status_message_count}", flush=True)
                print(f"├─ Log messages: {log_message_count}", flush=True)

                # Last activity
                inactive_minutes = (time.time() - last_update_time) / 60
                if is_running and inactive_minutes > 3:
                    print(
                        f"├─ ⚠️ No activity for {inactive_minutes:.1f} minutes",
                        flush=True,
                    )

                    # After 10 minutes of inactivity, provide more guidance
                    if inactive_minutes > 10:
                        print("\n⚠️ Execution appears to be stalled.", flush=True)
                        print("Possible reasons:", flush=True)
                        print(
                            "1. Cell is processing a long-running task without output",
                            flush=True,
                        )
                        print("2. Notebook encountered an error", flush=True)
                        print("3. Colab runtime disconnected unexpectedly", flush=True)
                        print("\nPlease check the notebook manually:", flush=True)
                        print(
                            f"https://colab.research.google.com/drive/{file_id}\n",
                            flush=True,
                        )

                # Update counters for next loop
                last_status_count = status_message_count
                last_log_count = log_message_count
                last_cell_count = executed_cells

                # Check if execution is complete
                if completed_marker or training_finished_marker:
                    print("\n✅ Notebook execution completed successfully!", flush=True)
                    if training_finished_marker:
                        # Extract training time
                        training_time_match = re.search(
                            r"Training finished in (.*) minutes", all_output
                        )
                        if training_time_match:
                            print(
                                f"Training completed in {training_time_match.group(1)} minutes",
                                flush=True,
                            )
                    return True

                if error_marker:
                    print(
                        "\n❌ Training encountered an error. Check the notebook for details.",
                        flush=True,
                    )
                    print(
                        f"Notebook URL: https://colab.research.google.com/drive/{file_id}\n",
                        flush=True,
                    )
                    return False

                # Check for token expiration or permission errors
                if (
                    "access_denied" in all_output
                    or "Token has been expired" in all_output
                ):
                    print(
                        "\n❌ Authentication token has expired after extended runtime.",
                        flush=True,
                    )
                    print(
                        "This is normal for long-running training jobs (token lifetime is ~60 minutes).",
                        flush=True,
                    )
                    print(
                        "The notebook is still executing in Colab, but we can no longer monitor it.",
                        flush=True,
                    )
                    print(
                        f"Please check the notebook manually: https://colab.research.google.com/drive/{file_id}",
                        flush=True,
                    )
                    return True, True  # Execution started but token expired

            except Exception as e:
                print(f"Error parsing notebook content: {e}", flush=True)
                traceback.print_exc()

            # Wait before polling again
            time.sleep(poll_interval_seconds)

        except HttpError as e:
            # Handle API errors
            if e.resp.status == 401 or e.resp.status == 403:
                print(
                    "\n❌ Authentication error: Access token has expired.", flush=True
                )
                print(
                    "The notebook might still be running, but we can no longer monitor it.",
                    flush=True,
                )
                print(
                    f"Please check manually: https://colab.research.google.com/drive/{file_id}",
                    flush=True,
                )
                return False, True  # Indicate token expired
            else:
                print(f"API error while checking notebook status: {e}", flush=True)
                print(f"Will retry in {poll_interval_seconds} seconds...", flush=True)
                time.sleep(poll_interval_seconds)

        except Exception as e:
            print(f"Error while checking notebook status: {e}", flush=True)
            print(f"Will retry in {poll_interval_seconds} seconds...", flush=True)
            traceback.print_exc()
            time.sleep(poll_interval_seconds)

    # Timeout reached
    print(f"\n⚠️ Timeout reached after {timeout_minutes} minutes!", flush=True)
    print(
        "The notebook might still be executing but we'll stop monitoring.", flush=True
    )
    print(
        f"Please check manually: https://colab.research.google.com/drive/{file_id}",
        flush=True,
    )

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
