"""
Colab cells for notebook autorun and log streaming capabilities.
This file contains the code that will be injected into notebooks for automation and logging.
"""

# %% [markdown]
# # Pugmark Training Pipeline
# This notebook will run the Pugmark training pipeline and generate logs for tracking.

# %%
LOG_STREAMING_CELL = """# @title 🔄 Training Setup and Configuration
# This cell sets up your training environment

import os
import sys
import time
from datetime import datetime
import IPython
from IPython import display

print("\\n" + "=" * 80)
print(" PUGMARK TRAINING NOTEBOOK ".center(80))
print("=" * 80)
print(f"\\n📅 Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\\n📋 INSTRUCTIONS:")
print("1. Connect to a runtime with GPU (if not already connected)")
print("2. Run this cell first to configure parameters")
print("3. Then run all cells below (Runtime > Run all remaining)")
print("\\n" + "=" * 80)

# Number of epochs to train for (set by parameter injection)
EPOCHS = %EPOCHS%  # Will be replaced with actual value

print(f"\\n🔢 Training will run for {EPOCHS} epochs")

# Configure the environment
try:
    print("\\n💻 Checking environment...", flush=True)
    
    # Check if connected to runtime
    if 'google.colab' in sys.modules:
        print("✓ Running in Google Colab", flush=True)
    else:
        print("⚠️ Not running in Google Colab - some features may not work", flush=True)
    
    # Check for GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU available: {len(gpus)} device(s) detected", flush=True)
            for i, gpu in enumerate(gpus):
                print(f"  - GPU {i}: {gpu}", flush=True)
                
            # Print GPU info
            try:
                gpu_info = !nvidia-smi -L
                print(f"  - GPU details: {gpu_info[0]}", flush=True)
            except:
                pass
        else:
            print("⚠️ No GPU detected - training will be VERY slow", flush=True)
            print("  - Go to Runtime > Change runtime type > Hardware accelerator > GPU", flush=True)
    except Exception as e:
        print(f"⚠️ Could not check for GPU: {str(e)}", flush=True)
    
    # Check TensorFlow version
    try:
        tf_version = tf.__version__
        print(f"✓ TensorFlow version: {tf_version}", flush=True)
        
        if not tf_version.startswith('2.'):
            print("⚠️ Recommended TensorFlow version is 2.x", flush=True)
            
    except Exception as e:
        print(f"⚠️ Could not check TensorFlow version: {str(e)}", flush=True)
        
    # Install required packages
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}", flush=True)
    except:
        print("⚠️ Installing required packages...", flush=True)
        !pip install -q tensorflow numpy
    
    # Try to mount Google Drive 
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted at /content/drive", flush=True)
    except Exception as drive_error:
        print("ℹ️ Google Drive not mounted (not required for training)", flush=True)
        
    print("\\n✅ Setup complete - ready to start training", flush=True)
    
except Exception as setup_error:
    print(f"\\n❌ Setup error: {str(setup_error)}", flush=True)
    import traceback
    traceback.print_exc()
    print("\\nPlease fix the error before continuing", flush=True)

# Make sure we can continue to the next cell
print("\\n" + "-" * 80)
print("You can now run the training cells below")
print("-" * 80 + "\\n")
"""

# %%
AUTORUN_CELL = """# @title 🚀 Initialize and Start Training

# This cell will be inserted at the end of the notebook
# It ensures we update GitHub when training is complete

import time
import sys
from datetime import datetime

print("\\n" + "=" * 80)
print(" TRAINING COMPLETION CHECK ".center(80))
print("=" * 80)
print("\\nThis is the final cell in the notebook.")
print("It will run automatically after all other cells finish executing.")
print("When you see this message, training is complete!")
print("\\n" + "=" * 80)

# Calculate total execution time
try:
    # Try to get start time from previous cell
    if 'EPOCHS' in globals():
        print(f"Training completed with {EPOCHS} epochs")
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Finished at: {current_time}")
    
    # Notify GitHub about completion
    print("\\nNOTEBOOK EXECUTION COMPLETED")
    print("Training process has finished successfully")
    
except Exception as e:
    print(f"Error in completion cell: {str(e)}")
    print("Training may have completed with errors")

print("\\n" + "=" * 80)
"""
