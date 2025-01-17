# Start from the base Python image
FROM python:3.11-slim AS builder

# Install dependencies and tools
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --verbose

# Copy the application code
COPY src src/

# Set PYTHONPATH for application imports
ENV PYTHONPATH="/src"

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
