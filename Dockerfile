# Use Google's pre-built GPU image
FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12.py310:latest

# Install Python dependencies
RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    google-cloud-storage

# Copy training code
COPY train.py /train.py

# Entrypoint for Vertex AI
ENTRYPOINT ["python", "/train.py"]