import os
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import pytest
import torch
from twitter_classification.data import TextDataset, preprocess, preprocess_data
from tests import _PATH_RAW_DATA

# Skip the test if raw data is missing
@pytest.mark.skipif(not os.path.exists(_PATH_RAW_DATA), reason="Raw data not found")
def test_data_loading():
    """Test if data loads correctly from CSV."""
    dataset = TextDataset(data_dir=_PATH_RAW_DATA, file_name="train.csv")
    assert len(dataset) > 0, "Dataset should have more than 0 entries"

    sample = dataset[0]
    assert isinstance(sample["text"], str), "Text should be a string"
    if "label" in sample:
        assert isinstance(sample["label"], torch.Tensor), "Label should be a tensor"


def test_preprocess_function():
    """Test preprocessing of a single text sample."""
    raw_text = "This is a tweet! @user http://example.com"
    processed_text = preprocess(raw_text)
    assert isinstance(processed_text, str), "Processed text should be a string"
    assert "http" in processed_text, "URLs should be normalized to 'http'"


@patch("twitter_classification.data.TextDataset")
@patch("pandas.read_csv")
def test_preprocess_data(mock_read_csv, mock_text_dataset):
    """Test the preprocessing function on the whole dataset."""
    # Mock the dataset
    mock_read_csv.return_value = pd.DataFrame({"text": ["This is a test"]})
    mock_text_dataset.return_value = [{"text": "This is a test"}]

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    preprocess_data(raw_dir, processed_dir)

    mock_read_csv.assert_called()
