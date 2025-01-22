FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY ./requirements_frontend.txt /app/requirements_frontend.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements_frontend.txt

# Copy the frontend application
COPY ./src/twitter_classification/frontend.py /app/frontend.py

# Expose port 8080 (required by Cloud Run)
EXPOSE 8080

# Command to start the Streamlit app on port 8080
ENTRYPOINT ["streamlit", "run", "frontend.py", "--server.port", "8080", "--server.address=0.0.0.0"]

