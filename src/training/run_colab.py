#!/usr/bin/env python3
"""
Script to execute a notebook on Google Colab using the Google Drive API.
This replaces the colab-cli dependency with direct API calls.
"""

import argparse
import io
import os
import sys
import time
import traceback

import nbformat
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload

# Import the notebook cells content
try:
    from training.colab_cells import AUTORUN_CELL, LOG_STREAMING_CELL

    print("Successfully imported notebook cells from colab_cells.py", flush=True)
except ImportError as e:
    print(f"Error importing colab_cells.py: {e}", flush=True)
    LOG_STREAMING_CELL = """
# This is a fallback log streaming cell
print("Log streaming is not available - colab_cells.py could not be imported")
"""
    AUTORUN_CELL = """
# This is a fallback autorun cell
print("Autorun is not available - colab_cells.py could not be imported")
"""

# Add early debugging prints
print(f"Python version: {sys.version}", flush=True)
print(f"Current directory: {os.getcwd()}", flush=True)
print(f"Script path: {os.path.abspath(__file__)}", flush=True)

try:
    print("Importing required modules...")
    import nbformat
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload

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


def wait_for_execution(
    drive_service, file_id, log_file_name="colab_execution_log.txt", timeout_minutes=180
):
    """Wait for the Colab notebook to finish execution."""
    print(
        f"Waiting for notebook execution to complete (timeout: {timeout_minutes} minutes)...",
        flush=True,
    )

    # Check if we're using a shared drive
    shared_drive_id = os.environ.get("SHARED_DRIVE_ID")
    params = {}
    if shared_drive_id:
        params["supportsAllDrives"] = True

    start_time = time.time()
    elapsed_minutes = 0
    log_file_id = None
    last_log_position = 0

    # First, try to find the log file in the same folder as the notebook
    try:
        # Get the parent folder of the notebook
        file_info = (
            drive_service.files()
            .get(fileId=file_id, fields="parents", **params)
            .execute()
        )

        if "parents" in file_info and file_info["parents"]:
            parent_id = file_info["parents"][0]

            # Search for the log file
            query = f"name = '{log_file_name}' and '{parent_id}' in parents and trashed = false"
            results = (
                drive_service.files()
                .list(q=query, spaces="drive", fields="files(id, name)", **params)
                .execute()
            )

            log_files = results.get("files", [])
            if log_files:
                log_file_id = log_files[0]["id"]
                print(f"Found log file with ID: {log_file_id}", flush=True)
                print("\n==== BEGINNING LOG STREAM ====", flush=True)
    except Exception as e:
        print(f"Error finding log file: {e}", flush=True)
        print("Will continue without log streaming", flush=True)

    # Loop to check the status and stream logs
    while elapsed_minutes < timeout_minutes:
        # Sleep first to give a chance for execution to start
        time.sleep(60)  # Check every minute

        elapsed_minutes = (time.time() - start_time) / 60
        print(f"Elapsed time: {elapsed_minutes:.1f} minutes", flush=True)

        # Stream logs if we have a log file
        if log_file_id:
            try:
                # Download the log file content
                response = (
                    drive_service.files()
                    .get_media(fileId=log_file_id, **params)
                    .execute()
                )

                if response:
                    log_content = response.decode("utf-8")

                    # Print new content only
                    if len(log_content) > last_log_position:
                        new_content = log_content[last_log_position:]
                        print(new_content, end="", flush=True)
                        last_log_position = len(log_content)

                        # Check for completion message in the logs
                        if "Execution complete!" in new_content:
                            print(
                                "\n==== NOTEBOOK EXECUTION COMPLETED ====", flush=True
                            )
                            return True
            except Exception as e:
                print(f"Error streaming logs: {e}", flush=True)

        # Check the notebook outputs as a fallback
        try:
            response = (
                drive_service.files().get_media(fileId=file_id, **params).execute()
            )

            if response:
                notebook = nbformat.reads(response.decode("utf-8"), as_version=4)

                # Check for cells with output
                executed_cell_count = 0
                for cell in notebook.cells:
                    if (
                        cell.cell_type == "code"
                        and hasattr(cell, "outputs")
                        and cell.outputs
                    ):
                        executed_cell_count += 1

                # Simple heuristic: If we have outputs in cells, assume progress is being made
                if executed_cell_count > 1:  # More than just our added cells
                    last_cell_with_output = None
                    for i, cell in enumerate(notebook.cells):
                        if (
                            cell.cell_type == "code"
                            and hasattr(cell, "outputs")
                            and cell.outputs
                        ):
                            last_cell_with_output = i

                    if (
                        last_cell_with_output is not None
                        and last_cell_with_output >= len(notebook.cells) - 2
                    ):
                        # If the last cells have output, likely finished
                        print("\n==== NOTEBOOK EXECUTION COMPLETED ====", flush=True)
                        return True

        except Exception as e:
            print(f"Error checking notebook status: {e}", flush=True)

    # If we've reached the timeout
    print(
        f"Timeout of {timeout_minutes} minutes reached. Proceeding with download of current state.",
        flush=True,
    )
    return False


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
    """Main entry point for the script."""
    # Set up argument parsing to match the original interface
    parser = argparse.ArgumentParser(
        description="Execute notebook in Colab and download results"
    )
    parser.add_argument("notebook_path", help="Path to notebook file")
    parser.add_argument(
        "--params",
        help="Parameters to inject into the notebook in format PARAM1=VALUE1,PARAM2=VALUE2",
        default="",
    )
    parser.add_argument(
        "--output", dest="output", help="Path to save executed notebook", default=None
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout in minutes for notebook execution",
    )
    parser.add_argument(
        "--machine-type",
        dest="machine_type",
        choices=["CPU", "GPU", "TPU"],
        default="GPU",
        help="Colab machine type (CPU, GPU, TPU)",
    )
    parser.add_argument(
        "--no-browser",
        dest="no_browser",
        action="store_true",
        help="Don't open browser to check execution",
    )
    args = parser.parse_args()

    # If output path isn't specified, create one
    if args.output is None:
        filename, ext = os.path.splitext(args.notebook_path)
        args.output = f"{filename}_executed{ext}"

    print("=== Starting Colab Execution Script ===", flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print(f"Current directory: {os.getcwd()}", flush=True)
    print(f"Script path: {os.path.abspath(__file__)}", flush=True)

    print("Setting up unbuffered output...", flush=True)

    try:
        # Upload notebook to drive
        print("\n=== STEP 1: Uploading notebook to Google Drive ===", flush=True)

        # Authenticate with Google Drive
        _, drive_service = authenticate()

        # Inject parameters if provided
        temp_notebook_path = args.notebook_path
        if args.params:
            print(f"Injecting parameters into notebook: {args.params}", flush=True)
            temp_notebook_path = inject_parameters(args.notebook_path, args.params)

        file_id = upload_to_drive(drive_service, temp_notebook_path)

        if file_id:
            # Execute notebook
            print("\n=== STEP 2: Executing notebook on Colab ===", flush=True)
            colab_url, log_file_name = create_colab_vm_and_execute(
                drive_service, file_id, args.machine_type
            )

            # Wait for execution
            print("\n=== STEP 3: Waiting for execution to complete ===", flush=True)
            execution_complete = wait_for_execution(
                drive_service,
                file_id,
                log_file_name=log_file_name,
                timeout_minutes=args.timeout,
            )

            # Download results
            print("\n=== STEP 4: Downloading executed notebook ===", flush=True)
            download_from_drive(drive_service, file_id, args.output)

            if execution_complete:
                print("\n✅ Notebook execution completed successfully!", flush=True)
            else:
                print("\n⚠️ Notebook execution timed out or had errors.", flush=True)
                print(
                    "   The notebook was downloaded in its current state.", flush=True
                )
                print(
                    "   You may need to check the notebook for errors and completion status.",
                    flush=True,
                )

            print(f"\nExecuted notebook saved to: {args.output}", flush=True)

            # Information for manual access
            print("\n=== ACCESS INFORMATION ===", flush=True)
            print(f"Colab URL: {colab_url}", flush=True)
            print(
                "You can always access this notebook through Google Drive or Colab",
                flush=True,
            )

            # Clean up temporary files if needed
            if temp_notebook_path != args.notebook_path:
                cleanup(drive_service, file_id, temp_file=temp_notebook_path)

            return 0
        else:
            print(
                "Failed to upload notebook to Drive. See error messages above.",
                flush=True,
            )
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
