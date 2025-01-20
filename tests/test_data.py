import os
import pytest
import pandas as pd
import torch 
from pathlib import Path
from twitter_classification.data import TextDataset, preprocess_data, preprocess
from tests import _PATH_RAW_DATA, _PATH_PROCESSED_DATA

# Skip the test if raw data is missing
@pytest.mark.skipif(not os.path.exists(_PATH_RAW_DATA), reason="Raw data not found")
def test_data_loading():
    """Test if data loads correctly from CSV."""
    # Assume 'train.csv' exists in raw data folder
    dataset = TextDataset(data_dir=_PATH_RAW_DATA, file_name="train.csv")
    assert len(dataset) > 0, "Dataset should have more than 0 entries"

    # Test if each sample is of expected shape, assuming it's text data
    sample = dataset[0]
    assert isinstance(sample['text'], str), "Text should be a string"

    if 'label' in sample:
        assert isinstance(sample['label'], torch.Tensor), "Label should be a tensor"

# Check if the preprocessing function works as expected
def test_preprocess_function():
    """Test preprocessing of a single text sample."""
    raw_text = "This is a tweet! @user http://example.com"
    processed_text = preprocess(raw_text)
    assert isinstance(processed_text, str), "Processed text should be a string"
    assert "http" in processed_text, "URLs should be normalized to 'http'"

# Check if preprocessing for a dataset works as expected
@pytest.mark.skipif(not os.path.exists(_PATH_RAW_DATA), reason="Raw data not found")
def test_preprocess_data():
    """Test the preprocessing function on the whole dataset."""
    # Convert raw_dir and processed_dir to Path objects
    raw_dir = Path(_PATH_RAW_DATA)
    processed_dir = Path(_PATH_PROCESSED_DATA)

    # Run the preprocessing on raw data
    preprocess_data(raw_dir=raw_dir, processed_dir=processed_dir)

    # Check that processed files are saved in the processed directory
    processed_file = processed_dir / "train_processed.csv"
    assert processed_file.exists(), f"Processed file {processed_file} not found"
    
    # Check if some basic changes have occurred (e.g., empty 'text' column)
    data = pd.read_csv(processed_file)
    assert not data['text'].isnull().any(), "Text column should not have any null values after preprocessing"

# Check that labels are represented in the dataset
@pytest.mark.skipif(not os.path.exists(_PATH_RAW_DATA), reason="Raw data not found")
def test_labels_representation():
    """Test that all labels are represented in the dataset."""
    dataset = TextDataset(data_dir=_PATH_RAW_DATA, file_name="train.csv")
    if "target" in dataset.data.columns:
        labels = dataset.data["target"].unique()
        assert len(labels) > 0, "No labels found in the dataset"
        # Optional: Check if all labels are represented
        assert all(label in labels for label in range(len(labels))), "Not all labels are represented in the dataset"


# To calculate the code coverage
# pip install coverage
#coverage run -m pytest tests/
#coverage report
