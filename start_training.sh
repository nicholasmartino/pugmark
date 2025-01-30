# Set project ID (only needed once per session)
export PROJECT_ID=$(gcloud config get-value project)
export REGION=us-central1
export REPO_NAME=pugmark
export IMAGE_NAME=footprint-pix2pix-trainer
export IMAGE_TAG=latest

# Set image URI variable
export IMAGE_URI=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}

# Authenticate Docker
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Verify authentication
gcloud auth list  # Should show active account
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build with your Dockerfile
docker build -t ${IMAGE_URI} . --progress=plain

# Push to Artifact Registry
docker push ${IMAGE_URI}

# List images in your repository
gcloud artifacts packages list \
  --repository=$REPO_NAME \
  --location=$REGION

# Check specific tags
gcloud artifacts tags list \
  --package=$IMAGE_NAME \
  --repository=$REPO_NAME \
  --location=$REGION

# 1. List existing repositories
gcloud artifacts repositories list --location=${REGION}

# 2. If missing, create the repository
gcloud artifacts repositories create pugmark \
  --repository-format=docker \
  --location=${REGION} \
  --project=${PROJECT_ID}

# 3. Rebuild and push with proper URI
docker build -t ${IMAGE_URI} .
docker push ${IMAGE_URI}

# 4. Verify image exists
gcloud artifacts packages list \
  --repository=pugmark \
  --location=${REGION} \
  --project=${PROJECT_ID}

gcloud ai-platform jobs submit training pix2pix_$(date +"%Y%m%d_%H%M%S") \
    --region=${REGION} \
    --master-image-uri=${IMAGE_URI} \
    --scale-tier=CUSTOM \
    --master-machine-type=n1-standard-4 \
    --master-accelerator=type=nvidia-tesla-t4,count=1 \
    --job-dir=gs://metro-vancouver-regional-district/jobs \
    --stream-logs 
