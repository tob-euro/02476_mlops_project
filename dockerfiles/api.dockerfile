FROM python:3.11-slim AS builder

# Install system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the entire project into the container
COPY . /app

# Install project dependencies using setup.py
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Set PYTHONPATH for application imports
ENV PYTHONPATH="/app/src"

# Expose port 8000 for external access
EXPOSE 8000

# Add a health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Switch to a non-root user
RUN useradd -ms /bin/bash appuser
USER appuser

# Command to run the application
ENTRYPOINT ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "src.twitter_classification.api:app"]
