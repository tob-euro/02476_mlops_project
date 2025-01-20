import os
import pytest
import torch
from transformers import RobertaForSequenceClassification
from twitter_classification.model import build_model
from twitter_classification.train import train_model
from tests import _PATH_MODELS, _PATH_RAW_DATA, _PATH_PROCESSED_DATA

# Skip the test if model directory is missing
@pytest.mark.skipif(not os.path.exists(_PATH_MODELS), reason="Model directory not found")
def test_build_model():
    """Test if the model builds correctly."""
    model = build_model()
    assert isinstance(model, RobertaForSequenceClassification), "Model should be an instance of RobertaForSequenceClassification"
    assert model.config.num_labels == 2, "Model should have 2 output labels"

# To calculate the code coverage
# pip install coverage
#coverage run -m pytest tests/
#coverage report
