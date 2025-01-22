import torch
from unittest.mock import patch, MagicMock
from transformers import AutoModelForSequenceClassification, BertConfig, BertForSequenceClassification
from twitter_classification.model import build_model

def test_build_model_with_mock():
    """Test if the model builds correctly using a mocked model."""
    # Mock the model and its config
    mock_model = MagicMock(spec=AutoModelForSequenceClassification)
    mock_model.config = MagicMock()
    mock_model.config.num_labels = 2

    with patch("twitter_classification.model.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model):
        model = build_model(model_name="cardiffnlp/twitter-roberta-base")

        assert isinstance(model, AutoModelForSequenceClassification)
        assert model.config.num_labels == 2

def test_model_initialization_without_weights():
    """Test model initialization with a local configuration."""
    config = BertConfig(num_labels=2)
    model = BertForSequenceClassification(config)
    input_ids = torch.randint(0, 1000, (4, 128))  # Batch size 4, sequence length 128
    outputs = model(input_ids)
    assert outputs.logits.shape == torch.Size([4, 2]), "Logits should have the correct shape"
