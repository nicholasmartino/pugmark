"""
Colab cells for notebook autorun and log streaming capabilities.
This file contains the code that will be injected into notebooks for automation and logging.
"""

LOG_STREAMING_CELL = """
# Cell for streaming logs
import io
import sys
import time
import threading
import json
import google.colab
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# Log file name - this will be used to stream logs back to the CI process
LOG_FILE_NAME = "colab_execution_log.txt"
STATUS_FILE_NAME = "colab_execution_status.json"

# Create files in Drive to store logs and status
def setup_log_streaming():
    # Authenticate for Drive access
    auth.authenticate_user()
    
    # Get parent folder ID (same as this notebook)
    drive_service = build('drive', 'v3', cache_discovery=False)
    file_info = drive_service.files().get(
        fileId=google.colab.drive.get_file_id(),
        fields='parents'
    ).execute()
    
    parent_id = file_info.get('parents', ['root'])[0]
    
    # Create log file in the same folder
    file_metadata = {
        'name': LOG_FILE_NAME,
        'mimeType': 'text/plain',
        'parents': [parent_id]
    }
    
    log_file = drive_service.files().create(
        body=file_metadata,
        fields='id'
    ).execute()
    
    log_file_id = log_file.get('id')
    print(f"Created log file with ID: {log_file_id}")
    
    # Create status file for cell execution progress
    status_metadata = {
        'name': STATUS_FILE_NAME,
        'mimeType': 'application/json',
        'parents': [parent_id]
    }
    
    status_file = drive_service.files().create(
        body=status_metadata,
        fields='id'
    ).execute()
    
    status_file_id = status_file.get('id')
    print(f"Created status file with ID: {status_file_id}")
    
    # Initialize status file
    init_status = {
        'total_cells': 0,  # Will be updated when execution starts
        'current_cell': 0,
        'status': 'initializing',
        'last_update': time.time(),
        'error': None
    }
    
    update_status_file(drive_service, status_file_id, init_status)
    
    return log_file_id, status_file_id, drive_service

# Function to update status file
def update_status_file(drive_service, file_id, status_data):
    media = MediaIoBaseUpload(
        io.BytesIO(json.dumps(status_data).encode()),
        mimetype='application/json',
        resumable=True
    )
    
    drive_service.files().update(
        fileId=file_id,
        media_body=media
    ).execute()

# Setup log streaming
log_file_id, status_file_id, log_drive_service = setup_log_streaming()

# Redirect stdout to also write to the log file
class LogRedirector:
    def __init__(self, file_id, drive_service):
        self.terminal = sys.stdout
        self.file_id = file_id
        self.drive_service = drive_service
        self.buffer = io.StringIO()
        self.last_upload = time.time()
        self.upload_interval = 3  # Upload every 3 seconds
        
    def write(self, message):
        self.terminal.write(message)
        self.buffer.write(message)
        
        # Upload if it's been more than upload_interval seconds
        if time.time() - self.last_upload > self.upload_interval:
            self.flush()
            
    def flush(self):
        content = self.buffer.getvalue()
        if content:
            try:
                # Upload to the log file
                media = MediaIoBaseUpload(
                    io.BytesIO(content.encode()),
                    mimetype='text/plain',
                    resumable=True
                )
                
                self.drive_service.files().update(
                    fileId=self.file_id,
                    media_body=media
                ).execute()
                
                self.last_upload = time.time()
                self.buffer = io.StringIO()
            except Exception as e:
                self.terminal.write(f"Error uploading logs: {e}\\n")

# Start log redirection
log_redirector = LogRedirector(log_file_id, log_drive_service)
sys.stdout = log_redirector

# Update status object
notebook_status = {
    'total_cells': 0,  # Will be set by autorun cell
    'current_cell': 0,
    'status': 'ready',
    'last_update': time.time(),
    'error': None
}

# Start a background thread to periodically flush logs
def periodic_flush():
    while True:
        time.sleep(5)
        try:
            print("--- PERIODIC LOG UPDATE ---")
            sys.stdout.flush()
            
            # Also update status to show we're still alive
            notebook_status['last_update'] = time.time()
            update_status_file(log_drive_service, status_file_id, notebook_status)
            
        except Exception as e:
            print(f"Error in periodic flush: {e}")
        
threading.Thread(target=periodic_flush, daemon=True).start()

print(f"LOG STREAMING SETUP COMPLETE - Logs will be written to file ID: {log_file_id}")
print(f"STATUS TRACKING SETUP COMPLETE - Status will be written to file ID: {status_file_id}")

# Make status file ID available to the autorun cell
status_file_global = status_file_id
log_drive_service_global = log_drive_service
"""

AUTORUN_CELL = """
# Auto-execution cell added by CI/CD
import IPython
import time
import sys
import json
import traceback

# Get status file ID from previous cell
status_file_id = status_file_global
drive_service = log_drive_service_global

# Print runtime info
print("=== RUNTIME INFORMATION ===")
!nvidia-smi  # Show GPU info if available
!python --version  # Show Python version
print("===========================")

# Function to update status
def update_status(status):
    try:
        from googleapiclient.http import MediaIoBaseUpload
        import io
        
        media = MediaIoBaseUpload(
            io.BytesIO(json.dumps(status).encode()),
            mimetype='application/json',
            resumable=True
        )
        
        drive_service.files().update(
            fileId=status_file_id,
            media_body=media
        ).execute()
    except Exception as e:
        print(f"Error updating status: {e}")

# Get all cells in the notebook
notebook = IPython.get_ipython().kernel.shell.user_ns['_ih']
cells = [cell for i, cell in notebook.items() if i != 0 and isinstance(i, int) and cell.strip()]

# Skip the first two cells (the log redirector and this autorun cell)
cells_to_run = cells[2:] if len(cells) > 2 else []

# Update status with total cell count
notebook_status['total_cells'] = len(cells_to_run)
notebook_status['status'] = 'running'
update_status(notebook_status)

print(f"Found {len(cells_to_run)} cells to execute")

# Function to run all cells one by one
def run_cells_one_by_one():
    print("Starting cell-by-cell execution in 5 seconds...")
    time.sleep(5)
    
    # Make sure we're connected to the runtime
    IPython.get_ipython().run_cell("from google.colab import runtime; runtime.connect()")
    print("Connected to runtime. Starting execution...")
    
    for i, cell_code in enumerate(cells_to_run):
        cell_num = i + 1  # 1-indexed for display
        
        # Update status before executing cell
        notebook_status['current_cell'] = cell_num
        notebook_status['status'] = f'executing_cell_{cell_num}'
        notebook_status['last_update'] = time.time()
        update_status(notebook_status)
        
        print(f"\\n==== EXECUTING CELL {cell_num}/{len(cells_to_run)} ====")
        print(f"Cell preview: {cell_code[:100]}...")  # Show first 100 chars
        
        # Execute the cell
        try:
            start_time = time.time()
            IPython.get_ipython().run_cell(cell_code)
            execution_time = time.time() - start_time
            print(f"\\n==== CELL {cell_num}/{len(cells_to_run)} COMPLETED in {execution_time:.2f}s ====")
            
            # Update status after successful execution
            notebook_status['status'] = f'completed_cell_{cell_num}'
            notebook_status['last_update'] = time.time()
            update_status(notebook_status)
            
            # Force flush to ensure logs are written
            sys.stdout.flush()
            
        except Exception as e:
            error_msg = f"Error executing cell {cell_num}: {str(e)}\\n{traceback.format_exc()}"
            print(f"\\n==== ERROR IN CELL {cell_num}/{len(cells_to_run)} ====")
            print(error_msg)
            
            # Update status with error
            notebook_status['status'] = 'error'
            notebook_status['error'] = error_msg
            notebook_status['last_update'] = time.time()
            update_status(notebook_status)
            
            # Continue with next cell despite error
            print("Continuing with next cell...")
    
    # Update final status
    notebook_status['status'] = 'completed'
    notebook_status['last_update'] = time.time()
    update_status(notebook_status)
    print("\\n==== NOTEBOOK EXECUTION COMPLETED ====")

# Run automatically
run_cells_one_by_one()
"""
