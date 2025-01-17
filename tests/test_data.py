from torch.utils.data import Dataset
from twitter_classification.data import TextDataset, clean_text


def test_clean_text():
    """Test the clean_text function."""
    print("[INFO] Testing clean_text...")
    text = "Hello, @world! Check out this link: https://www.example.com"
    cleaned = clean_text(text)
    expected = "hello, user! check out this link: url"
    assert cleaned == expected, f"Expected: {expected}, but got: {cleaned}"
    print("[PASS] clean_text works as expected.")


def test_text_dataset():
    """Test the TextDataset class for train and test data."""
    print("[INFO] Testing TextDataset...")

    # Test train.csv
    try:
        train_dataset = TextDataset("data/raw", file_name="train.csv")
        assert len(train_dataset) > 0, "Train dataset should not be empty."
        train_sample = train_dataset[0]
        assert "text" in train_sample, "Train sample should contain 'text' key."
        assert "label" in train_sample, "Train sample should contain 'label' key."
        assert train_sample["label"] in [0, 1], "Train sample label should be 0 or 1."
        print(f"[INFO] Train dataset sample: {train_sample}")
    except Exception as e:
        print(f"[FAIL] Train dataset test failed: {e}")
        raise

    # Test test.csv
    try:
        test_dataset = TextDataset("data/raw", file_name="test.csv")
        assert len(test_dataset) > 0, "Test dataset should not be empty."
        test_sample = test_dataset[0]
        assert "text" in test_sample, "Test sample should contain 'text' key."
        assert "label" not in test_sample, "Test sample should not contain 'label' key."
        print(f"[INFO] Test dataset sample: {test_sample}")
    except Exception as e:
        print(f"[FAIL] Test dataset test failed: {e}")
        raise

    print("[PASS] TextDataset works as expected for both train and test datasets.")



if __name__ == "__main__":
    print("[INFO] Running tests...")
    test_clean_text()
    test_text_dataset()
    print("[INFO] All tests completed successfully.")
