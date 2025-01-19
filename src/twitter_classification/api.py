from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path

# Initialize FastAPI app
app = FastAPI()

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

@app.on_event("shutdown")
async def cleanup():
    """Clean up resources during shutdown."""
    global model, tokenizer
    del model, tokenizer
    print("Cleaned up resources.")

# Root endpoint
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

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Perform inference on the input text and return a label and confidence score.

    Args:
        request (PredictionRequest): Input text for classification.

    Returns:
        PredictionResponse: Predicted label and confidence score.
    """
    try:
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
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
