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

AUTORUN_CELL = """
# Cell-by-cell execution with detailed output
import IPython
import time
import sys
import traceback
import os
import datetime

# Function to print important status messages
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

# Print runtime info with very visible banner
print(f"\\n{'#' * 80}")
print("#" + " COLAB RUNTIME INFORMATION ".center(78) + "#")
print(f"{'#' * 80}\\n")
sys.stdout.flush()

!nvidia-smi  # Show GPU info if available
!python --version  # Show Python version
!hostname  # Show hostname
!df -h  # Show disk space

# Get all cells in the notebook
def get_notebook_cells():
    try:
        shell = IPython.get_ipython().kernel.shell
        notebook = shell.user_ns['_ih']
        # Convert to a list, skipping non-integer indexes and empty cells
        cells = []
        for i in sorted([i for i in notebook.keys() if isinstance(i, int)]):
            if i > 0 and notebook[i].strip():  # Skip cell 0 and empty cells
                cells.append(notebook[i])
        
        # Skip the first two injected cells (logging and this autorun cell)
        return cells[2:] if len(cells) > 2 else []
    except Exception as e:
        print(f"Error getting notebook cells: {str(e)}\\n{traceback.format_exc()}")
        return []

# Function to run all cells one by one
def run_cells_one_by_one():
    log_status("Preparing to execute notebook cell by cell", important=True)
    
    # Print additional info to help diagnose issues
    print(f"\\n{'=' * 50}")
    print(f"IPython version: {IPython.__version__}")
    print(f"Running in directory: {os.getcwd()}")
    print(f"{'=' * 50}\\n")
    
    time.sleep(2)
    
    # Make sure we're connected to the runtime
    try:
        IPython.get_ipython().run_cell("from google.colab import runtime; runtime.connect()")
        log_status("Connected to Colab runtime")
    except Exception as e:
        log_status(f"Error connecting to runtime: {str(e)}", important=True)
    
    # Get cells to execute
    cells_to_run = get_notebook_cells()
    log_status(f"Found {len(cells_to_run)} cells to execute", important=True)
    
    # Update global execution status
    execution_status["total_cells"] = len(cells_to_run)
    execution_status["current_status"] = "running"
    
    # Print cell count information
    print(f"\\n{'=' * 50}")
    print(f"TOTAL CELLS TO EXECUTE: {len(cells_to_run)}")
    print(f"{'=' * 50}\\n")
    sys.stdout.flush()
    
    cell_execution_times = []
    
    for i, cell_code in enumerate(cells_to_run):
        cell_num = i + 1  # 1-indexed for display
        
        # Update status before executing cell
        execution_status["current_cell"] = cell_num
        execution_status["current_status"] = f"executing_cell_{cell_num}"
        
        # Print a very visible cell execution marker
        print(f"\\n{'▼' * 80}")
        print(f"▼▼▼ EXECUTING CELL {cell_num}/{len(cells_to_run)} ▼▼▼")
        print(f"{'▼' * 80}")
        
        # Print a truncated preview of the cell for debugging
        preview = cell_code.replace('\\n', ' ')[:100] + ('...' if len(cell_code) > 100 else '')
        print(f"Cell content preview: {preview}")
        sys.stdout.flush()
        
        # Execute the cell
        try:
            start_time = time.time()
            IPython.get_ipython().run_cell(cell_code)
            execution_time = time.time() - start_time
            cell_execution_times.append(execution_time)
            
            # Print a very visible cell completion marker
            print(f"\\n{'▲' * 80}")
            print(f"▲▲▲ CELL {cell_num}/{len(cells_to_run)} COMPLETED in {execution_time:.2f}s ▲▲▲")
            print(f"{'▲' * 80}\\n")
            sys.stdout.flush()
            
            # Update status after successful execution
            execution_status["current_status"] = f"completed_cell_{cell_num}"
            
            # Force flush
            sys.stdout.flush()
            
        except Exception as e:
            error_msg = f"Error executing cell {cell_num}: {str(e)}\\n{traceback.format_exc()}"
            
            # Print a very visible error marker
            print(f"\\n{'!' * 80}")
            print(f"!!! ERROR IN CELL {cell_num}/{len(cells_to_run)} !!!")
            print(error_msg)
            print(f"{'!' * 80}\\n")
            sys.stdout.flush()
            
            # Update status with error
            execution_status["current_status"] = "error"
            execution_status["error"] = error_msg
            
            # Continue with next cell despite error
            log_status("Continuing with next cell...")
    
    # Calculate and print execution summary
    if cell_execution_times:
        total_time = sum(cell_execution_times)
        avg_time = total_time / len(cell_execution_times)
        
        print(f"\\n{'#' * 80}")
        print(f"# EXECUTION SUMMARY ".ljust(79, '#'))
        print(f"# Total execution time: {total_time:.2f}s".ljust(79, '#'))
        print(f"# Average cell execution time: {avg_time:.2f}s".ljust(79, '#'))
        print(f"# Fastest cell: {min(cell_execution_times):.2f}s".ljust(79, '#'))
        print(f"# Slowest cell: {max(cell_execution_times):.2f}s".ljust(79, '#'))
        print(f"{'#' * 80}\\n")
        sys.stdout.flush()
    
    # Update final status
    execution_status["current_status"] = "completed"
    
    # Print completion banner
    print(f"\\n{'*' * 80}")
    print("*" + " " * 78 + "*")
    print("*" + " NOTEBOOK EXECUTION COMPLETED ".center(78) + "*")
    print("*" + " " * 78 + "*")
    print(f"{'*' * 80}\\n")
    sys.stdout.flush()

# Run automatically with error handling
try:
    run_cells_one_by_one()
except Exception as e:
    # Print a very visible critical error banner
    print(f"\\n{'!' * 80}")
    print("!" + " CRITICAL ERROR IN AUTORUN ".center(78, '!') + "!")
    print(f"{str(e)}")
    print(traceback.format_exc())
    print(f"{'!' * 80}\\n")
    sys.stdout.flush()
"""
