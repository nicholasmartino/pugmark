export PROJECT_NAME="pugmark"
export SERVICE_ACCOUNT="github-service-account"
export STORAGE_BUCKET="gs://metro-vancouver-regional-district"
export WORKLOAD_PROVIDER="github-identity-provider"

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
else
    echo ".env file not found"
    exit 1
fi

gcloud services enable iamcredentials.googleapis.com \
  --project "${GCP_PROJECT_ID}"

gcloud services enable secretmanager.googleapis.com \
    --project "${GCP_PROJECT_ID}"

gcloud artifacts repositories create "${PROJECT_NAME}" \
  --repository-format=docker \
  --location=us-central1 \
  --project=${GCP_PROJECT_ID}


### 
# IDENTITY PROVIDER & SERVICE ACCOUNT
###

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

export WORKLOAD_IDENTITY_POOL_ID=$(gcloud iam workload-identity-pools describe "github-actions-pool" \
  --project="${GCP_PROJECT_ID}" \
  --location="global" \
  --format="value(name)")

# Create GitHub Actions service account
gcloud iam service-accounts create "${SERVICE_ACCOUNT}" \
  --display-name "GitHub Actions Service Account"

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

gcloud iam service-accounts add-iam-policy-binding "${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --project="${GCP_PROJECT_ID}" \
  --role="roles/run.admin" \
  --member="principalSet://iam.googleapis.com/${WORKLOAD_IDENTITY_POOL_ID}/attribute.repository/${GITHUB_REPO}"

gcloud iam workload-identity-pools providers describe "${WORKLOAD_PROVIDER}" \
  --project="${GCP_PROJECT_ID}" \
  --location="global" \
  --workload-identity-pool="github-actions-pool" \
  --format="value(name)"


###
# PROJECTS
###

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.admin"

# Check it wokred
gcloud projects get-iam-policy $GCP_PROJECT_ID \
    --flatten="bindings[].members" \
    --format='table(bindings.role)' \
    --filter="bindings.members:${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

# Grant Cloud Run Admin role to the service account
gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/run.admin"

# Grant additional required roles
gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
  --member="principalSet://iam.googleapis.com/${WORKLOAD_IDENTITY_POOL_ID}/attribute.repository/${GITHUB_REPO}" \
  --role="roles/run.admin"

# Grant Cloud Run Invoker role to both service account and workload identity
gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/run.invoker"

gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
  --member="principalSet://iam.googleapis.com/${WORKLOAD_IDENTITY_POOL_ID}/attribute.repository/${GITHUB_REPO}" \
  --role="roles/run.invoker"

# Grant Storage Admin role to your service account
gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# If you want more granular permissions, use:
gcloud storage buckets add-iam-policy-binding "${STORAGE_BUCKET}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"


###
# SECRETS
###

# First, create the secret in Secret Manager
gcloud secrets create "${GCP_SECRET_NAME}" \
    --project="${GCP_PROJECT_ID}"

# Then, add the service account key JSON as the secret value
gcloud secrets versions add "${GCP_SECRET_NAME}" \
    --project="${GCP_PROJECT_ID}" \
    --data-file="${GCP_SECRET_PATH}"
