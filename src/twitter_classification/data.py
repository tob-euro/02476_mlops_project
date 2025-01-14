import os
import re
import zipfile
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from torch.utils.data import Dataset
from loguru import logger

# Paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Configure logging
logger.add(
    "logs/data_pipeline.log",
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    level="INFO",
    rotation="1 MB",
    compression="zip",
)


class TextDataset(Dataset):
    """Custom Dataset for loading text data from CSV files."""

    def __init__(self, data_dir: str, file_name: str = "train.csv"):
        """Initialize the dataset by loading a CSV file."""
        csv_path = Path(data_dir) / file_name
        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")
        self.data = pd.read_csv(csv_path)
        self.has_labels = "target" in self.data.columns  # Check if 'target' column exists

    def __len__(self):
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve a sample by index."""
        row = self.data.iloc[idx]
        sample = {"text": row.get("text", "")}
        if self.has_labels:
            sample["label"] = row.get("target", None)
        return sample



def download_data(raw_dir: Path) -> None:
    """Download and extract data from Kaggle."""
    logger.info("Downloading dataset from Kaggle...")

    load_dotenv()  # Load .env variables
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    if not kaggle_username or not kaggle_key:
        raise EnvironmentError("Kaggle API credentials are missing. Check .env file.")

    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", "nlp-getting-started", "-p", str(raw_dir)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Kaggle download failed: {e}")

    # Extract dataset
    zip_path = raw_dir / "nlp-getting-started.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        zip_path.unlink()
    logger.info("Dataset downloaded and extracted successfully.")


def clean_text(text: str) -> str:
    """Clean and normalize text for transformer-based models."""
    if pd.isna(text):
        return ""

    patterns = [
        (r"http\S+|www\S+|https\S+", "<URL>"),
        (r"@\w+", "<USER>"),
        (r"\s+", " "),
    ]
    text = text.lower().strip()
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return text


def preprocess_data(raw_dir: Path, processed_dir: Path) -> None:
    """Preprocess raw data and save it to the processed directory."""
    logger.info("Preprocessing data...")
    processed_dir.mkdir(parents=True, exist_ok=True)

    for file_name in ["train.csv", "test.csv"]:
        data = pd.read_csv(raw_dir / file_name)
        data["text"] = data["text"].fillna("").apply(clean_text)
        data.to_csv(processed_dir / f"{file_name.split('.')[0]}_processed.csv", index=False)
    logger.info("Preprocessing complete.")


def main():
    """Run the data pipeline."""
    download_data(RAW_DIR)
    preprocess_data(RAW_DIR, PROCESSED_DIR)


if __name__ == "__main__":
    logger.info("Starting data pipeline...")
    main()
    logger.info("Data pipeline complete.")
