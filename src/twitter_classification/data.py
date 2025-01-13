import os
import zipfile
import subprocess
import pandas as pd
import logging
from pathlib import Path
import re
from dotenv import load_dotenv

# Define paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

def download_data(raw_dir: Path) -> None:
    """Download data from Kaggle using the CLI."""
    logging.info("Downloading dataset from Kaggle...")

    # Load .env file
    load_dotenv()

    # Ensure credentials are loaded
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")
    if not kaggle_username or not kaggle_key:
        raise EnvironmentError(
            "Kaggle API credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY "
            "in a .env file or as environment variables."
        )

    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", "nlp-getting-started", "-p", str(raw_dir)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logging.error(
            "Error downloading the dataset. Ensure you have joined the competition at "
            "https://www.kaggle.com/c/nlp-getting-started. Details: %s", str(e)
        )
        raise

    # Extract ZIP file
    zip_path = raw_dir / "nlp-getting-started.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        zip_path.unlink()  # Remove the ZIP file after extraction
    logging.info("Dataset downloaded and extracted successfully.")

def clean_text(text: str) -> str:
    """Clean and normalize text for transformer-based models."""
    if pd.isna(text):
        return ""

    # Define regex patterns
    patterns = [
        (r"http\S+|www\S+|https\S+", "<URL>"),  # URLs
        (r"@\w+", "<USER>"),                   # Mentions
        (r"\s+", " "),                         # Excessive whitespace
    ]
    
    text = text.lower().strip()
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return text

def preprocess_data(raw_dir: Path, processed_dir: Path) -> None:
    """Preprocess raw data and save it to the processed directory."""
    logging.info("Preprocessing data...")

    # Load raw data
    train_data = pd.read_csv(raw_dir / "train.csv")
    test_data = pd.read_csv(raw_dir / "test.csv")

    # Preprocess text column
    for dataset, name in zip([train_data, test_data], ["train", "test"]):
        dataset["text"] = dataset["text"].fillna("").apply(clean_text)
        dataset.to_csv(processed_dir / f"{name}_processed.csv", index=False)

    logging.info("Data preprocessing complete. Processed data saved.")

def main(raw_dir: str = "data/raw", processed_dir: str = "data/processed") -> None:
    """Main function to orchestrate data downloading and preprocessing."""
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    download_data(raw_path)
    preprocess_data(raw_path, processed_path)

if __name__ == "__main__":
    setup_logging()
    main()
