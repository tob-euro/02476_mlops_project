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

# Copy the entire project into the container
COPY . /app

# Expose port 8000 for the application
EXPOSE 8000

# Command to start the FastAPI app
CMD ["uvicorn", "src.twitter_classification.api:app", "--host", "0.0.0.0", "--port", "8000"]
