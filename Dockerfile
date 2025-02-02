# Use Google's pre-built GPU image
FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12.py310:latest

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy training code
COPY train.py /train.py

# Entrypoint for Vertex AI
ENTRYPOINT ["python", "/train.py"]