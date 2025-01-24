from fastapi.testclient import TestClient
from twitter_classification.api import app

# Create a TestClient instance
client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Welcome to the Twitter Disaster Classification API!" in data["message"]

def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_metrics_endpoint():
    """Test the Prometheus metrics endpoint."""
    response = client.get("/metrics/")
    assert response.status_code == 200
    assert "prediction_requests" in response.text
