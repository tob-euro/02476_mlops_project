import os
import pytest
from transformers import RobertaForSequenceClassification, AutoModelForSequenceClassification
from twitter_classification.model import build_model

# Path to the specific model directory
_PATH_MODELS = "models/bert_disaster_tweets"

@pytest.mark.skipif(not os.path.exists(_PATH_MODELS), reason="Model directory not found")
def test_build_model_from_directory():
    """Test if the model builds correctly from a valid directory."""
    model = build_model(model_dir=_PATH_MODELS)
    assert isinstance(model, RobertaForSequenceClassification), (
        "The returned model should be a subclass of RobertaForSequenceClassification."
    )

def test_build_model_with_pretrained_name():
    """Test if the model builds correctly using a pretrained model name."""
    model = build_model(model_name="cardiffnlp/twitter-roberta-base")
    # Check if the model is an instance of RobertaForSequenceClassification
    assert isinstance(model, RobertaForSequenceClassification), (
        "The returned model should be an instance of RobertaForSequenceClassification."
    )
    # Validate the number of output labels
    assert model.config.num_labels == 2, "The number of labels should match the config (2)."