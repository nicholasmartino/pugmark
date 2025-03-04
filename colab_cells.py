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
import google.colab
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# Log file name - this will be used to stream logs back to the CI process
LOG_FILE_NAME = "colab_execution_log.txt"

# Create a file in Drive to store logs
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
    
    return log_file_id, drive_service

# Setup log streaming
log_file_id, log_drive_service = setup_log_streaming()

# Redirect stdout to also write to the log file
class LogRedirector:
    def __init__(self, file_id, drive_service):
        self.terminal = sys.stdout
        self.file_id = file_id
        self.drive_service = drive_service
        self.buffer = io.StringIO()
        self.last_upload = time.time()
        self.upload_interval = 5  # Upload every 5 seconds
        
    def write(self, message):
        self.terminal.write(message)
        self.buffer.write(message)
        
        # Upload if it's been more than upload_interval seconds
        if time.time() - self.last_upload > self.upload_interval:
            self.flush()
            
    def flush(self):
        content = self.buffer.getvalue()
        if content:
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

# Start log redirection
log_redirector = LogRedirector(log_file_id, log_drive_service)
sys.stdout = log_redirector

# Start a background thread to periodically flush logs
def periodic_flush():
    while True:
        time.sleep(10)
        print("--- PERIODIC LOG UPDATE ---")
        sys.stdout.flush()
        
threading.Thread(target=periodic_flush, daemon=True).start()

print(f"LOG STREAMING SETUP COMPLETE - Logs will be written to file ID: {log_file_id}")
"""

AUTORUN_CELL = """
# Auto-execution cell added by CI/CD
import IPython
import time

# Print runtime info
print("=== RUNTIME INFORMATION ===")
!nvidia-smi  # Show GPU info if available
!python --version  # Show Python version
print("===========================")

# Function to run all cells
def run_all():
    print("Auto-executing all cells in 5 seconds...")
    time.sleep(5)
    IPython.get_ipython().run_cell("from google.colab import runtime; runtime.connect()")
    print("Connected to runtime. Starting execution...")
    IPython.get_ipython().run_line_magic("run", "-i '{path}' 2:999")  # Skip the first cell (this one)
    print("Execution complete!")

# Run automatically
run_all()
"""
