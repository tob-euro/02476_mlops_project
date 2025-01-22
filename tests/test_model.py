import os
import pytest
from transformers import RobertaForSequenceClassification
from twitter_classification.model import build_model
from tests import _PATH_MODELS

# Skip the test if the model directory is missing
@pytest.mark.skipif(not os.path.exists(_PATH_MODELS), reason="Model directory not found")
def test_build_model_from_directory():
    """Test if the model builds correctly from a valid directory."""
    model = build_model(model_dir=_PATH_MODELS)
    assert isinstance(model, RobertaForSequenceClassification), (
        "The returned model should be an instance of RobertaForSequenceClassification."
    )
    assert model.config.num_labels == 2, "The number of labels should match the config (2)."


def test_build_model_with_pretrained_name():
    """Test if the model builds correctly using a pretrained model name."""
    model_name = "cardiffnlp/twitter-roberta-base"
    model = build_model(model_name=model_name)
    assert isinstance(model, RobertaForSequenceClassification), (
        "The returned model should be an instance of RobertaForSequenceClassification."
    )
    assert model.config.num_labels == 2, "The number of labels should match the config (2)."