#!/bin/bash

# Deploy using latest image
gcloud run deploy pugmark-service \
  --image us-central1-docker.pkg.dev/$GCP_PROJECT_ID/pugmark/pugmark:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --project $GCP_PROJECT_ID \
  --service-account=github-service-account@$GCP_PROJECT_ID.iam.gserviceaccount.com
  