# Building footprint generator

Building footprint generator based on the [pix2pix](https://www.tensorflow.org/tutorials/generative/pix2pix) model using conditional Generative Adversarial Networks. Training data was sourced from Statistics Canada's [Open Database of Buildings]('https://www.statcan.gc.ca/eng/lode/databases/odb').

```bash
# Authenticate with your Google account
gcloud auth login

# Set default project
gcloud config set project YOUR_PROJECT_ID

# Configure Docker credential helper
gcloud auth configure-docker

# Enable necessary APIs
gcloud services enable \
    containerregistry.googleapis.com \
    aiplatform.googleapis.com \
    cloudbuild.googleapis.com

# Make script executableD
chmod +x start_training.sh

# Run the submission script
./start_training.sh
```

## Results

![](https://raw.githubusercontent.com/nicholas-martino/pix2pix/master/footprints_gen/150epochs/fg3.png)
![](https://raw.githubusercontent.com/nicholas-martino/pix2pix/master/footprints_gen/150epochs/fg1.png)
![](https://raw.githubusercontent.com/nicholas-martino/pix2pix/master/footprints_gen/150epochs/fg4.png)

## License

[![cc-by-image](https://i.creativecommons.org/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)
