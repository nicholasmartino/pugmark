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

## Results

![](https://raw.githubusercontent.com/nicholasmartino/pugmar/master/footprints_gen/150epochs/fg3.png)
![](https://raw.githubusercontent.com/nicholasmartino/pugmar/master/footprints_gen/150epochs/fg1.png)
![](https://raw.githubusercontent.com/nicholasmartino/pugmark/master/footprints_gen/150epochs/fg4.png)

## License

[![cc-by-image](https://i.creativecommons.org/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)
