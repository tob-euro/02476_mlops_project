FROM python:3.11-slim AS base

# Install system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the entire project into the container
COPY . /app

# Install project dependencies using setup.py
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Environment variables for training
ENV DATASET_PATH=data/processed/train_processed.csv
ENV MODEL_NAME=bert-base-uncased
ENV OUTPUT_DIR=models/bert_disaster_tweets

# Default entry point for training
ENTRYPOINT ["python", "-u", "src/twitter_classification/train.py"]
