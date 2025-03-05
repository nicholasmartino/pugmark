# Building footprint generator

Building footprint generator based on the [pix2pix](https://www.tensorflow.org/tutorials/generative/pix2pix) model using conditional Generative Adversarial Networks. Training data was sourced from Statistics Canada's [Open Database of Buildings]('https://www.statcan.gc.ca/eng/lode/databases/odb').

## Setup Options

### Option 1: Local and Cloud Run Setup

1. Connect GitHub repository on Google Cloud console

2. Run the following setup scripts

```bash
# Authenticate with your Google account
gcloud auth login

chmod +x scripts/gcloud_setup.sh
chmod +x scripts/gcloud_submit_job.sh

./setup_gcloud.sh
```

### Option 2: Google Colab Notebook

1. Open the Colab notebook in the `notebooks` directory:
   - [train_footprints.ipynb](notebooks/train_footprints.ipynb)

2. Run the notebook interactively in Google Colab
   - Click the "Open in Colab" button at the top of the notebook
   - The notebook will clone the repository and set up the environment

### Option 3: Automated Training via GitHub Actions (Recommended)

The project supports two methods for automated training through GitHub Actions:

#### Vertex AI Training (Recommended)
- Uses Google Cloud's Vertex AI platform for reliable GPU training
- Provides better monitoring and more stable execution
- Executes notebooks with full GPU acceleration
- No token expiration issues during long training runs
- Run the workflow from the Actions tab or trigger automatically on pushes to master

#### Colab GPU Training (Deprecated)
- The Colab-based training workflow is now deprecated
- While still functional, it may experience limitations with token expiration
- Will be removed in a future release

To use the automated training:
- Any changes to the notebook or training code in the `src/training` directory will trigger automatic execution
- Executed notebooks are saved as artifacts and can be downloaded from the Actions tab
- You can also manually trigger training with custom parameters from the Actions tab

## Training on Vertex AI with GPU

This project includes a GitHub Actions workflow to run training jobs on Google Cloud's Vertex AI platform with GPU acceleration.

### Prerequisites

1. A Google Cloud project with Vertex AI API enabled
2. A service account with the following roles:
   - Vertex AI User
   - Storage Admin
   - Artifact Registry Reader
3. Workload Identity Federation configured for GitHub Actions
4. A Google Cloud Storage bucket for storing training outputs
5. A Docker image in Google Artifact Registry

### Running the Training Job

1. Go to the "Actions" tab in your GitHub repository
2. Select the "Vertex AI GPU Training" workflow
3. Click "Run workflow"
4. Configure the following parameters:
   - **Machine Type**: Select the VM machine type (e.g., n1-standard-8)
   - **Accelerator Type**: Select the GPU type (e.g., NVIDIA_TESLA_T4)
   - **Accelerator Count**: Select the number of GPUs (1-8)
5. Click "Run workflow" to start the training job

### Monitoring the Training Job

Once the workflow is triggered, you can monitor the training job in the Google Cloud Console:

1. Go to the [Vertex AI Custom Jobs page](https://console.cloud.google.com/vertex-ai/training/custom-jobs)
2. Find your job with the prefix "pugmark-training-"
3. Click on the job to view details, logs, and metrics

### Accessing Training Outputs

Training outputs are stored in Google Cloud Storage at:
```
gs://{PROJECT_ID}-ml/pugmark/training/{TIMESTAMP}
```

This includes TensorBoard logs, checkpoints, and any other outputs from the training process.

## Training on Vertex AI Workbench

This project supports running training directly on a Vertex AI Workbench instance with GPU acceleration through GitHub Actions. The workflow can automatically create a new instance or use an existing one.

### Prerequisites

1. A Google Cloud project with the following APIs enabled:
   - Vertex AI API
   - Compute Engine API
   - Notebooks API

2. Configure necessary IAM permissions for your service account:
   - Compute Instance Admin
   - Service Account User
   - Storage Admin
   - Vertex AI User
   - Notebooks Admin

### Running the Training Job

1. Go to the "Actions" tab in your GitHub repository
2. Select the "Vertex AI Workbench Training" workflow
3. Click "Run workflow"
4. Configure the following parameters:
   - **Instance Name**: Name to use for the Workbench instance
   - **Create Instance**: Whether to create a new instance if it doesn't exist (default: true)
   - **Machine Type**: VM machine type for the instance
   - **Accelerator Type**: GPU type to use
   - **Accelerator Count**: Number of GPUs
   - **Delete After Training**: Whether to delete the instance after training completes
   - **Epochs**: Number of training epochs (optional)
   - **Batch Size**: Training batch size (optional)
5. Click "Run workflow" to start the training job

### How It Works

This approach:
1. Uses GitHub Actions to authenticate with Google Cloud
2. Creates a new Vertex AI Workbench instance with GPU (if requested) or connects to an existing one
3. Clones the repository on the instance
4. Runs the training code directly on the GPU-enabled instance
5. Saves outputs to Google Cloud Storage
6. Optionally deletes the instance after training completes

### Advantages Over Custom Training Jobs

- No Docker container required
- Easier debugging and interactive development
- Dynamic instance creation and management
- Full customization of hardware specifications
- Optional automatic cleanup after training
- Can use the Workbench instance for other tasks when not training
- Simpler workflow and quicker startup

### Monitoring the Training Job

For the Vertex AI Workbench training approach, you can monitor your job in several ways:

1. **View logs directly on the Workbench instance**:
   - Go to the [Vertex AI Workbench instances page](https://console.cloud.google.com/vertex-ai/workbench/instances)
   - Click on the instance running your training
   - Open JupyterLab (Open JupyterLab button)
   - Use the terminal to check logs: `cat /tmp/training_completed.flag` to check if training has completed

2. **Monitor GPU usage and system metrics**:
   - Go to the [Compute Engine instances page](https://console.cloud.google.com/compute/instances)
   - Find your Workbench instance (same name you provided in the workflow)
   - Click on the instance and go to the "Monitoring" tab

3. **Check training outputs in Google Cloud Storage**:
   - Outputs are stored at: `gs://{PROJECT_ID}-ml/pugmark/training/{TIMESTAMP}`
   - Access via the [Cloud Storage browser](https://console.cloud.google.com/storage/browser)

### Accessing Training Outputs

Training outputs are stored in Google Cloud Storage at:
```
gs://{PROJECT_ID}-ml/pugmark/training/{TIMESTAMP}
```

This includes TensorBoard logs, checkpoints, and any other outputs from the training process.

### Troubleshooting Workbench Training

If you encounter issues with the Workbench training approach, consider the following:

1. **SSH Connection Issues**:
   - Ensure the service account has correct IAM permissions for Compute Engine
   - Check that the instance is fully initialized before attempting to connect
   - If using a custom network, verify that SSH ports are open

2. **GPU Not Available**:
   - Confirm GPU driver installation with `nvidia-smi` command on the instance
   - Verify the selected zone has the requested GPU type available
   - Check GPU quotas in your Google Cloud project

3. **Training Fails to Start**:
   - Examine logs on the instance (connect via SSH or Jupyter terminal)
   - Check that dependencies are correctly installed
   - Verify environment variables are properly set

4. **Common Error Messages**:
   - "No CUDA-capable device is detected" - GPU driver installation issue
   - "SSH connection failed" - Instance not fully initialized or network issue
   - "Instance [name] does not exist" - Check instance name and region/zone

5. **Getting Support**:
   - For Vertex AI Workbench-specific issues, see [Google Cloud documentation](https://cloud.google.com/vertex-ai/docs/workbench/user-managed/troubleshooting)
   - For Pugmark-specific issues, create a GitHub issue in this repository

## Results

![](https://raw.githubusercontent.com/nicholasmartino/pugmar/master/footprints_gen/150epochs/fg3.png)
![](https://raw.githubusercontent.com/nicholasmartino/pugmar/master/footprints_gen/150epochs/fg1.png)
![](https://raw.githubusercontent.com/nicholasmartino/pugmark/master/footprints_gen/150epochs/fg4.png)

## License

[![cc-by-image](https://i.creativecommons.org/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)
