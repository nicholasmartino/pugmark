# Use Google's pre-built GPU image
FROM us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12.py310:latest

# Set working directory
WORKDIR /app

# Install Python dependencies with cache cleanup
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip && \
    rm -rf /tmp/*

# Copy training code
COPY training/ ./training/

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import os; exit(0 if os.path.exists('/tmp/healthy') else 1)"

# Entrypoint for Vertex AI
ENTRYPOINT ["python3", "training/train.py"]