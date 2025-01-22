import os
import pytest
from unittest.mock import patch, MagicMock
from transformers import RobertaForSequenceClassification
from twitter_classification.model import build_model
from tests import _PATH_MODELS

@patch("transformers.AutoModelForSequenceClassification.from_pretrained")
def test_build_model_from_directory(mock_from_pretrained):
    """Test if the model builds correctly from a valid directory."""
    # Mock the returned model
    mock_model = MagicMock(spec=RobertaForSequenceClassification)
    mock_from_pretrained.return_value = mock_model

    # Call the function under test
    model = build_model(model_dir=_PATH_MODELS)

    # Assertions
    assert model is mock_model, "The returned model should match the mock model."
    mock_from_pretrained.assert_called_once_with(_PATH_MODELS)


@patch("transformers.AutoModelForSequenceClassification.from_pretrained")
def test_build_model_with_pretrained_name(mock_from_pretrained):
    """Test if the model builds correctly using a pretrained model name."""
    # Mock the returned model
    model_name = "cardiffnlp/twitter-roberta-base"
    mock_model = MagicMock(spec=RobertaForSequenceClassification)
    mock_from_pretrained.return_value = mock_model

    # Call the function under test
    model = build_model(model_name=model_name)

    # Assertions
    assert model is mock_model, "The returned model should match the mock model."
    mock_from_pretrained.assert_called_once_with(
        model_name, num_labels=2
    )
