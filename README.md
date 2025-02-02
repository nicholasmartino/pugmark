# Building footprint generator

Building footprint generator based on the [pix2pix](https://www.tensorflow.org/tutorials/generative/pix2pix) model using conditional Generative Adversarial Networks. Training data was sourced from Statistics Canada's [Open Database of Buildings]('https://www.statcan.gc.ca/eng/lode/databases/odb').

1. Connect GitHub repository on Google Cloud console

2. Run the following setup scripts

```bash
# Authenticate with your Google account
gcloud auth login

# Create github actions pool
gcloud iam workload-identity-pools create "github-actions-pool" \
  --location="global" \
  --description="GitHub Actions pool" \
  --display-name="GitHub Actions"

# Then create the provider in the pool
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --location="global" \
  --workload-identity-pool="github-actions-pool" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
  --attribute-condition="attribute.repository == 'nicholasmartino/pugmark'" \
  --project=${PROJECT_ID}

# Get the provider an add to repo secret
gcloud iam workload-identity-pools providers list \
  --location="global" \
  --workload-identity-pool="github-actions-pool" \
  --project=${PROJECT_ID}

# Add permission to run cloud setup script
chmod +x setup_gcloud.sh

./setup_gcloud.sh
```

## Results

![](https://raw.githubusercontent.com/nicholas-martino/pix2pix/master/footprints_gen/150epochs/fg3.png)
![](https://raw.githubusercontent.com/nicholas-martino/pix2pix/master/footprints_gen/150epochs/fg1.png)
![](https://raw.githubusercontent.com/nicholas-martino/pix2pix/master/footprints_gen/150epochs/fg4.png)

## License

[![cc-by-image](https://i.creativecommons.org/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)
