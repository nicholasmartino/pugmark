"""
Colab cells for notebook autorun and log streaming capabilities.
This file contains the code that will be injected into notebooks for automation and logging.
"""

LOG_STREAMING_CELL = """
# Cell for direct output streaming
import io
import sys
import time
import threading
import json
import os
import datetime

# Set up custom logging that ensures output is visible and flushed immediately
class EnhancedOutput:
    def __init__(self):
        self.terminal = sys.stdout
        self.last_flush = time.time()
        self.buffer = ""
        self.flush_interval = 0.1  # Flush more frequently - every 0.1 seconds
        
    def write(self, message):
        self.terminal.write(message)
        self.buffer += message
        
        # Check if we should flush
        current_time = time.time()
        if current_time - self.last_flush > self.flush_interval:
            self.flush()
    
    def flush(self):
        if self.buffer:
            # Force Python to flush output
            self.terminal.flush()
            self.last_flush = time.time()
            self.buffer = ""

# Replace stdout with our enhanced version
sys.stdout = EnhancedOutput()

# Global execution status
execution_status = {
    "start_time": time.time(),
    "current_cell": 0,
    "total_cells": 0,
    "current_status": "initializing",
    "last_update": time.time(),
    "error": None
}

# Function to print status updates that will be visible in GitHub Actions logs
def log_status(message, important=False):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if important:
        separator = "=" * 40
        print(f"\\n{separator}")
        print(f"[COLAB_STATUS] {timestamp}: {message}")
        print(f"{separator}\\n")
    else:
        print(f"[COLAB_LOG] {timestamp}: {message}")
    sys.stdout.flush()

# Print a large banner to make output very visible
print(f"\\n{'*' * 80}")
print("*" + " " * 78 + "*")
print("*" + " COLAB EXECUTION STREAMING INITIALIZED ".center(78) + "*")
print("*" + " " * 78 + "*")
print(f"{'*' * 80}\\n")
sys.stdout.flush()

# Background thread to periodically print status updates
def status_reporter():
    while True:
        try:
            elapsed = time.time() - execution_status["start_time"]
            current = execution_status["current_cell"]
            total = execution_status["total_cells"]
            status = execution_status["current_status"]
            
            if total > 0:
                progress = f"{current}/{total} cells ({current/total*100:.1f}%)"
            else:
                progress = "Calculating..."
                
            log_status(f"Status: {status} | Progress: {progress} | Running for: {elapsed/60:.1f} minutes", important=True)
            
            # Update last_update to show we're still alive
            execution_status["last_update"] = time.time()
            
        except Exception as e:
            print(f"Error in status reporter: {str(e)}")
        
        time.sleep(15)  # Report status more frequently - every 15 seconds

# Start the status reporting thread
threading.Thread(target=status_reporter, daemon=True).start()

# Print initial status message
log_status("Output streaming initialized - logs will be visible in GitHub Actions", important=True)
log_status("This method uses direct output streaming instead of file-based logging")
"""

AUTORUN_CELL = '''# @title Autorun all cells sequentially {display-mode: "form"}
# This cell will execute all notebook cells in sequence

import time
import re
import json
import sys
import os
from IPython import display
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import Javascript

# Force runtime connection immediately
print("\\n" + "*" * 80)
print("* INITIALIZING RUNTIME AND VERIFYING CONNECTION *".center(78))
print("*" * 80 + "\\n")

# Aggressively try to connect to runtime and verify GPU
try:
    # Force runtime connection
    from google.colab import runtime
    runtime.connect(timeout_sec=60)  # Longer timeout to ensure connection
    print("✓ Successfully connected to runtime")
    
    # Try to mount drive
    from google.colab import drive
    try:
        drive.mount('/content/drive', force_remount=True)
        print("✓ Drive mounted successfully")
    except Exception as drive_error:
        print(f"Drive mounting not required or failed: {drive_error}")
    
    # Check for GPU
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU available: {len(gpus)} device(s)")
        # Try to get GPU info
        try:
            !nvidia-smi -L
        except:
            pass
    else:
        print("⚠️ No GPU found! Using CPU.")
    
    # Clear output for clean start
    display.clear_output(wait=True)
    print("✓ Runtime initialized successfully. Starting execution...", flush=True)
except Exception as init_error:
    print(f"Runtime initialization warning: {init_error}")
    print("Will continue with execution anyway...")

# Define globals for execution state
execution_state = {
    "start_time": time.time(),
    "current_cell": 0,
    "total_cells": 0,
    "status": "starting",
    "last_update": time.time(),
    "errors": []
}

def get_all_code_cells():
    """Get all code cells in the notebook, excluding this cell and the log cell"""
    # This uses IPython's internal API to get all cells
    shell = InteractiveShell.instance()
    cells = []
    
    # Loop through all cells
    for i, cell in enumerate(shell.user_ns.get('_ih', [])):
        # Skip empty cells, the autorun cell and the log cell
        if not cell or "Autorun all cells" in cell or "log streaming" in cell:
            continue
        cells.append((i, cell))
    
    execution_state["total_cells"] = len(cells)
    return cells

def print_status(msg, cell_num=None, total=None):
    """Print status message with formatting"""
    elapsed = (time.time() - execution_state["start_time"]) / 60
    if cell_num is not None and total is not None:
        # Print a special marker for parsing by the monitoring script
        print(f"\\n▼▼▼ EXECUTING CELL {cell_num}/{total} ▼▼▼", flush=True)
        # Print a preview of the cell content
        if current_cell_code:
            preview = current_cell_code.split("\\n")[0][:50]
            if len(preview) == 50:
                preview += "..."
            print(f"Cell preview: {preview}", flush=True)
    
    # Always include timestamp and elapsed time
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{elapsed:.1f}min] {msg}", flush=True)
    execution_state["last_update"] = time.time()
    execution_state["status"] = msg

def execute_cell(cell_num, code):
    """Execute a single cell and capture its output and errors"""
    global current_cell_code
    current_cell_code = code
    
    execution_state["current_cell"] = cell_num
    cell_start_time = time.time()
    
    # Report start of execution
    print_status(f"Executing cell {cell_num}/{execution_state['total_cells']}", 
                cell_num, execution_state["total_cells"])
    
    success = True
    try:
        # Execute the cell code
        shell = InteractiveShell.instance()
        result = shell.run_cell(code)
        
        if result.error_before_exec or result.error_in_exec:
            success = False
            error_msg = str(result.error_in_exec or result.error_before_exec)
            execution_state["errors"].append({
                "cell": cell_num,
                "error": error_msg
            })
            print(f"\\n❌ ERROR in cell {cell_num}: {error_msg}", flush=True)
    except Exception as e:
        success = False
        execution_state["errors"].append({
            "cell": cell_num,
            "error": str(e)
        })
        print(f"\\n❌ ERROR executing cell {cell_num}: {str(e)}", flush=True)
    
    # Calculate execution time
    exec_time = time.time() - cell_start_time
    exec_time_str = f"{exec_time:.1f}s" if exec_time < 60 else f"{exec_time/60:.1f}min"
    
    # Report completion
    status = "COMPLETED" if success else "FAILED"
    print(f"▲▲▲ CELL {cell_num}/{execution_state['total_cells']} {status} in {exec_time_str} ▲▲▲\\n", flush=True)
    
    # Return success/failure
    return success

# Start execution process
print("\\n" + "=" * 80, flush=True)
print("PUGMARK TRAINING AUTOMATION".center(80), flush=True)
print(f"Starting execution at {time.strftime('%Y-%m-%d %H:%M:%S')}".center(80), flush=True)
print("=" * 80 + "\\n", flush=True)

# Install required packages for runtime setup
print("Setting up runtime environment...", flush=True)
try:
    # Force runtime connection
    print("Checking runtime type...")
    import tensorflow as tf
    try:
        if tf.config.list_physical_devices('GPU'):
            print("✓ GPU is available")
            gpu_info = !nvidia-smi
            if gpu_info:
                print(f"GPU Info: {gpu_info[0]}")
        else:
            print("⚠️ No GPU found. Using CPU only.")
    except:
        print("Could not verify GPU. Will continue regardless.")
except:
    print("Could not import tensorflow yet. Continuing...")

try:
    # Run update and install packages first
    shell = InteractiveShell.instance()
    
    # Configure Colab environment for optimal training
    print("\\nConfiguring environment for training...", flush=True)
    setup_commands = [
        "pip install -q tensorflow==2.10.1 ipywidgets",
        "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')",
        "import numpy as np; print(f'NumPy version: {np.__version__}')",
        "import os; print(f'Environment: {os.environ.get(\"COLAB_GPU\", \"No GPU\")}')"
    ]
    
    for cmd in setup_commands:
        try:
            shell.run_cell(cmd)
        except Exception as e:
            print(f"Setup command failed: {e}", flush=True)
            # Continue anyway
            pass
    
    # Get all cells and execute them sequentially
    cells = get_all_code_cells()
    print(f"\\nFound {len(cells)} cells to execute", flush=True)
    current_cell_code = ""
    
    # Execute all cells
    for i, (cell_num, code) in enumerate(cells):
        # Skip the first cell (this auto-run cell)
        if "Autorun all cells" in code:
            continue
            
        success = execute_cell(cell_num, code)
        # Continue even if a cell fails
    
    # Final status
    errors = len(execution_state["errors"])
    if errors > 0:
        print(f"\\n⚠️ Execution completed with {errors} errors", flush=True)
    else:
        print("\\n✅ All cells executed successfully!", flush=True)
    
    # Report total execution time
    total_time = (time.time() - execution_state["start_time"]) / 60
    print(f"Total execution time: {total_time:.1f} minutes", flush=True)
    print("\\n" + "=" * 80, flush=True)
    print("NOTEBOOK EXECUTION COMPLETED".center(80), flush=True)
    print("=" * 80 + "\\n", flush=True)
    
except Exception as e:
    print(f"\\n❌ Fatal error in autorun: {str(e)}", flush=True)
    import traceback
    traceback.print_exc()
    print("\\nPlease check the error and run cells manually", flush=True)
'''
