# Use the base Python image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the requirements and install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Install DVC and required plugins
RUN pip install dvc[gdrive] dvc[gs]

# Copy the service account key file and set the Google Cloud credentials
COPY .env/mlops-448114-e0e59a05789f.json /app/mlops-448114-e0e59a05789f.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/mlops-448114-e0e59a05789f.json

# Copy the entire project into the container
COPY . /app

# Expose port 8080 for the application
EXPOSE 8080

# Command to pull DVC data and start the FastAPI app
CMD dvc pull && uvicorn src.twitter_classification.api:app --host 0.0.0.0 --port 8080
