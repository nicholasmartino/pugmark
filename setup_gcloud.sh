export SERVICE_ACCOUNT="github-service-account"
export WORKLOAD_PROVIDER="github-identity-provider"

gcloud services enable iamcredentials.googleapis.com \
  --project "${GCP_PROJECT_ID}"

# Create github actions pool
gcloud iam workload-identity-pools create "github-actions-pool" \
  --location="global" \
  --description="GitHub Actions pool" \
  --display-name="GitHub Actions"

# Then create the provider in the pool
gcloud iam workload-identity-pools providers create-oidc "${WORKLOAD_PROVIDER}" \
  --project="${GCP_PROJECT_ID}" \
  --location="global" \
  --workload-identity-pool="github-actions-pool" \
  --display-name="${WORKLOAD_PROVIDER}" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-condition="attribute.repository == '${GITHUB_REPO}'"

# Get the provider an add to repo secret
gcloud iam workload-identity-pools providers list \
  --location="global" \
  --workload-identity-pool="github-actions-pool" \
  --project=${GCP_PROJECT_ID}

# Create GitHub Actions service account
gcloud iam service-accounts create "${SERVICE_ACCOUNT}" \
  --display-name "GitHub Actions Service Account"

export WORKLOAD_IDENTITY_POOL_ID=$(gcloud iam workload-identity-pools describe "github-actions-pool" \
  --project="${GCP_PROJECT_ID}" \
  --location="global" \
  --format="value(name)")

# Grant IAM Role to GitHub Actions Identity
gcloud iam service-accounts add-iam-policy-binding "${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --project="${GCP_PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${WORKLOAD_IDENTITY_POOL_ID}/attribute.repository/${GITHUB_REPO}"

gcloud iam service-accounts add-iam-policy-binding "${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --project="${GCP_PROJECT_ID}" \
  --role="roles/artifactregistry.writer" \
  --member="principalSet://iam.googleapis.com/${WORKLOAD_IDENTITY_POOL_ID}/attribute.repository/${GITHUB_REPO}"

gcloud iam service-accounts add-iam-policy-binding "${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --project="${GCP_PROJECT_ID}" \
  --role="roles/artifactregistry.reader" \
  --member="principalSet://iam.googleapis.com/${WORKLOAD_IDENTITY_POOL_ID}/attribute.repository/${GITHUB_REPO}"

gcloud iam service-accounts add-iam-policy-binding "${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --project="${GCP_PROJECT_ID}" \
  --role="roles/run.developer" \
  --member="principalSet://iam.googleapis.com/${WORKLOAD_IDENTITY_POOL_ID}/attribute.repository/${GITHUB_REPO}"

gcloud iam workload-identity-pools providers describe "${WORKLOAD_PROVIDER}" \
  --project="${GCP_PROJECT_ID}" \
  --location="global" \
  --workload-identity-pool="github-actions-pool" \
  --format="value(name)"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.admin"

# Check it wokred
gcloud projects get-iam-policy $GCP_PROJECT_ID \
    --flatten="bindings[].members" \
    --format='table(bindings.role)' \
    --filter="bindings.members:${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
