FROM python:3.11-slim AS base

# Install system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for training
ENV DATASET_PATH=data/processed/train_processed.csv \
    MODEL_NAME=bert-base-uncased \
    OUTPUT_DIR=models/bert_disaster_tweets

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --verbose

# Copy the project files
COPY src/ src/
COPY configs/ configs/

# Expose default ports for potential debugging
EXPOSE 6006 8000

# Set entrypoint to training script
ENTRYPOINT ["python", "-u", "src/twitter_classification/train.py"]
