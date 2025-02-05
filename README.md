# Building footprint generator

Building footprint generator based on the [pix2pix](https://www.tensorflow.org/tutorials/generative/pix2pix) model using conditional Generative Adversarial Networks. Training data was sourced from Statistics Canada's [Open Database of Buildings]('https://www.statcan.gc.ca/eng/lode/databases/odb').

1. Connect GitHub repository on Google Cloud console

2. Run the following setup scripts

```bash
# Authenticate with your Google account
gcloud auth login

chmod +x setup_gcloud.sh

./setup_gcloud.sh

export PROJECT_NUMBER=$(gcloud projects describe ${GCP_PROJECT_ID} --format="value(projectNumber)")
gcloud iam service-accounts add-iam-policy-binding \
  github-actions@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-actions-pool/attribute.repository/${GITHUB_REPO}" \
  --role="roles/iam.workloadIdentityUser" \
  --project=${GCP_PROJECT_ID}

# Add Cloud Build Service Account role
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
  --member="serviceAccount:github-actions@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/cloudbuild.builds.builder"

# Grant Artifact Registry Writer role
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
  --member="serviceAccount:github-actions@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

# Grant Service Account User role
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
  --member="serviceAccount:github-actions@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# Add explicit repository permissions
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
  --member="serviceAccount:github-actions@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role=roles/owner

gcloud artifacts repositories add-iam-policy-binding pugmark \
  --location=us-central1 \
  --member=serviceAccount:github-actions@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
  --role=roles/artifactregistry.writer \
  --project=${GCP_PROJECT_ID}

gcloud artifacts repositories add-iam-policy-binding pugmark \
  --location=us-central1 \
  --member=serviceAccount:github-actions@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
  --role=roles/artifactregistry.repoAdmin \
  --project=${GCP_PROJECT_ID}

gcloud artifacts repositories add-iam-policy-binding pugmark \
  --location=us-central1 \
  --member=serviceAccount:github-actions@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
  --role=roles/artifactregistry.admin \
  --project=${GCP_PROJECT_ID}

gcloud artifacts repositories add-iam-policy-binding pugmark \
  --location=us-central1 \
  --member=serviceAccount:github-actions@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
  --role=roles/artifactregistry.containerRegistryMigrationAdmin \
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
