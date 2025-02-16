#!/bin/bash

# Delete job if exists (non-interactive)
gcloud run jobs delete pugmark-training \
  --project $GCP_PROJECT_ID \
  --region us-central1 \
  --quiet

# Submit training job
gcloud run jobs create pugmark-training \
  --image us-central1-docker.pkg.dev/$GCP_PROJECT_ID/pugmark/pugmark:latest \
  --region us-central1 \
  --project $GCP_PROJECT_ID \
  --service-account=github-service-account@$GCP_PROJECT_ID.iam.gserviceaccount.com \
  --task-timeout=43200 \
  --parallelism=1 \
  --cpu=4 \
  --memory=4G \
  --max-retries=0 \
  --command="python3" \
  --args="/app/src/training/train.py"

# Execute the job immediately after creation
gcloud run jobs execute pugmark-training \
  --project $GCP_PROJECT_ID \
  --region us-central1 \
  --wait
