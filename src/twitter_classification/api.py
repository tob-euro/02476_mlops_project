from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from prometheus_client import Counter, Histogram, Summary, make_asgi_app, CollectorRegistry
import torch
from pathlib import Path
import logging

# Initialize FastAPI app
app = FastAPI(strict_slashes=False)

# Global variables for the model and tokenizer
model = None
tokenizer = None
device = None

# Input schema
class PredictionRequest(BaseModel):
    text: str

# Output schema
class PredictionResponse(BaseModel):
    label: int
    confidence: float

# Define Prometheus metrics in a custom registry
MY_REGISTRY = CollectorRegistry()
error_counter = Counter(
    "prediction_error", "Number of prediction errors", registry=MY_REGISTRY
)
request_counter = Counter(
    "prediction_requests", "Number of prediction requests", registry=MY_REGISTRY
)
request_latency = Histogram(
    "prediction_latency_seconds", "Prediction latency in seconds", registry=MY_REGISTRY
)
text_length_summary = Summary(
    "text_length_summary", "Summary of text lengths in prediction requests", registry=MY_REGISTRY
)

# Expose the metrics endpoint
app.mount("/metrics/", make_asgi_app(registry=MY_REGISTRY))


@app.on_event("startup")
async def load_model():
    """Load the model and tokenizer during startup."""
    global model, tokenizer, device
    try:
        model_path = Path("models/bert_disaster_tweets")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

@app.get("/")
async def root():
    """Root endpoint with a friendly message."""
    return {
        "message": "Welcome to the Twitter Disaster Classification API! 🚀",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Perform inference on the input text and return a label and confidence score.

    Args:
        request (PredictionRequest): Input text for classification.

    Returns:
        PredictionResponse: Predicted label and confidence score.
    """
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty or whitespace.")

    request_counter.inc()  # Increment request counter
    with request_latency.time():  # Measure latency of request
        try:
            text_length_summary.observe(len(request.text))  # Observe text length

            # Tokenize the input text
            inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True).to(device)

            # Perform inference
            with torch.no_grad():
                outputs = model(**inputs)

            # Get predicted label and confidence
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, label = torch.max(probabilities, dim=1)

            return PredictionResponse(label=label.item(), confidence=confidence.item())
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            error_counter.inc()  # Increment error counter
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
