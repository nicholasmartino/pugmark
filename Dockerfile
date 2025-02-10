# Use Google's pre-built GPU image
FROM us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12.py310:latest

# Set working directory
WORKDIR /app

# Install Python dependencies with cache cleanup
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip && \
    rm -rf /tmp/*

# Create directory first
RUN mkdir -p /app/src/training
COPY src/training/* /app/src/training/

# Verify copy operation
RUN ls -lha /app/src/training/  # Changed to show contents of training directory

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import os; exit(0 if os.path.exists('/tmp/healthy') else 1)"
