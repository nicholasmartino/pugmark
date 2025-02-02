# Lightweight Python base
FROM python:3.10-slim

# Install system dependencies only needed for build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with cleanup
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    find /usr/local/lib -type d -name '__pycache__' -exec rm -rf {} + && \
    rm -rf /root/.cache

# Copy training code
COPY train.py /train.py

# Entrypoint for Vertex AI
ENTRYPOINT ["python", "/train.py"]