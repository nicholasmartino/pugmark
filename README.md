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
  --attribute-condition="attribute.repository == '${GITHUB_REPO}'" \
  --project=${GCP_PROJECT_ID}

# Get the provider an add to repo secret
gcloud iam workload-identity-pools providers list \
  --location="global" \
  --workload-identity-pool="github-actions-pool" \
  --project=${GCP_PROJECT_ID}

# Create GitHub Actions service account
gcloud iam service-accounts create github-actions \
  --display-name "GitHub Actions Service Account"

# Grant IAM Role to GitHub Actions Identity
export PROJECT_NUMBER=$(gcloud projects describe ${GCP_PROJECT_ID} --format="value(projectNumber)")
gcloud iam service-accounts add-iam-policy-binding \
  github-actions@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-actions-pool/attribute.repository/${GITHUB_REPO}" \
  --role="roles/iam.workloadIdentityUser" \
  --project=${GCP_PROJECT_ID}

# Grant Artifact Registry Writer role
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
  --member="serviceAccount:github-actions@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

# Grant Service Account User role
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
  --member="serviceAccount:github-actions@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# Add explicit repository permissions
gcloud artifacts repositories add-iam-policy-binding pugmark \
  --location=us-central1 \
  --member=serviceAccount:github-actions@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
  --role=roles/artifactregistry.writer \
  --project=${GCP_PROJECT_ID}

gcloud artifacts repositories add-iam-policy-binding pugmark \
  --location=us-central1 \
  --member=serviceAccount:github-actions@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
  --role=roles/artifactregistry.createOnPushWriter \
  --project=${GCP_PROJECT_ID}

gcloud artifacts repositories add-iam-policy-binding pugmark \
  --location=us-central1 \
  --member=serviceAccount:github-actions@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
  --role=roles/artifactregistry.createOnPushRepoAdmin \
  --project=${GCP_PROJECT_ID}
```

## Results

![](https://raw.githubusercontent.com/nicholasmartino/pugmar/master/footprints_gen/150epochs/fg3.png)
![](https://raw.githubusercontent.com/nicholasmartino/pugmar/master/footprints_gen/150epochs/fg1.png)
![](https://raw.githubusercontent.com/nicholasmartino/pugmark/master/footprints_gen/150epochs/fg4.png)

## License

[![cc-by-image](https://i.creativecommons.org/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)
