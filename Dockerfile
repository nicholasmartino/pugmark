# Use Google's pre-built GPU image
FROM us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12.py310:latest

# Install Python dependencies with cache cleanup
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip && \
    rm -rf /tmp/*

# Copy training code
COPY train.py /train.py

# Entrypoint for Vertex AI
ENTRYPOINT ["python", "/train.py"]