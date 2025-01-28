# Check container args are passed correctly
docker run gcr.io/pugmark-448918/footprint-pix2pix-training:latest --help

# Build the Docker image
docker build -t gcr.io/pugmark-448918/footprint-pix2pix-training:latest .

# Push to Google Container Registry
docker push gcr.io/pugmark-448918/footprint-pix2pix-training:latest

gcloud ai-platform jobs submit training pix2pix_$(date +"%Y%m%d_%H%M%S") \
    --region=us-central1 \
    --scale-tier=CUSTOM \
    --master-image-uri=gcr.io/pugmark-448918/footprint-pix2pix-training:latest \
    --master-machine-type=n1-standard-8 \
    --job-dir=gs://metro-vancouver-regional-district/jobs \
    --stream-logs 
