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

### Option 2: Google Colab Notebook (Recommended)

1. Open the Colab notebook in the `notebooks` directory:
   - [train_footprints.ipynb](notebooks/train_footprints.ipynb)

2. Run the notebook interactively in Google Colab
   - Click the "Open in Colab" button at the top of the notebook
   - The notebook will clone the repository and set up the environment

3. Automated Training via GitHub Actions
   - Any changes to the notebook or training code in the `src/training` directory will trigger automatic execution through GitHub Actions
   - Executed notebooks are saved as artifacts and can be downloaded from the Actions tab

## Results

![](https://raw.githubusercontent.com/nicholasmartino/pugmar/master/footprints_gen/150epochs/fg3.png)
![](https://raw.githubusercontent.com/nicholasmartino/pugmar/master/footprints_gen/150epochs/fg1.png)
![](https://raw.githubusercontent.com/nicholasmartino/pugmark/master/footprints_gen/150epochs/fg4.png)

## License

[![cc-by-image](https://i.creativecommons.org/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)
