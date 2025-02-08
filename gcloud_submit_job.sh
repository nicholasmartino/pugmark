#!/bin/bash

# Submit training job
gcloud run jobs create pugmark-training \
  --image us-central1-docker.pkg.dev/$GCP_PROJECT_ID/pugmark/pugmark:latest \
  --region us-central1 \
  --project $GCP_PROJECT_ID \
  --service-account=github-service-account@$GCP_PROJECT_ID.iam.gserviceaccount.com \
  --task-timeout=3600 \
  --parallelism=1 \
  --cpu=4 \
  --memory=4G \
  --command="/train.py" \
  