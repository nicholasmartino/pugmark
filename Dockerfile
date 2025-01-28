 # Base image with GPU support
FROM tensorflow/tensorflow:2.12.0-gpu

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    git \
    gcsfuse && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application files
COPY setup.py .
COPY pix2pix/ ./pix2pix/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e . && \
    pip install gcsfs google-cloud-storage

# Set entrypoint
ENTRYPOINT ["python", "pix2pix.py"]