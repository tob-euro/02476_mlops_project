import os
import re
import zipfile
import subprocess
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from dotenv import load_dotenv
from loguru import logger
import torch
import contractions

#halløj

# Import utility functions
from twitter_classification.utils import load_config, setup_logger

# Load configuration
config = load_config()

# Configure logging
setup_logger(
    log_file=config["logging"]["file"],
    level=config["logging"]["level"],
)

# Paths from config
RAW_DIR = Path(config["paths"]["raw_dir"])
PROCESSED_DIR = Path(config["paths"]["processed_dir"])


class TextDataset(Dataset):
    """
    Custom Dataset for loading text data from CSV files.

    Arguments:
        data_dir: string, the directory containing the CSV file.
        file_name: string, the name of the CSV file (default is 'train.csv').
    """

    def __init__(self, data_dir: str, file_name: str = "train.csv") -> None:
        csv_path = Path(data_dir) / file_name
        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")
        self.data = pd.read_csv(csv_path)
        self.has_labels = "target" in self.data.columns  # Check if 'target' column exists

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Retrieve a sample by index."""
        row = self.data.iloc[idx]
        sample = {"text": row.get("text", "")}
        if self.has_labels:
            sample["label"] = torch.tensor(row.get("target", 0), dtype=torch.long)
        return sample


def download_data(raw_dir: Path = RAW_DIR) -> None:
    """
    Download and extract data from Kaggle.

    Arguments:
        raw_dir: Path, the directory to download and extract the data.
    """
    logger.info("Downloading dataset from Kaggle...")

    # Load the .env/kaggle.env path from the config file
    kaggle_env_path = Path(config["paths"]["kaggle_env"])
    load_dotenv(dotenv_path=kaggle_env_path)

    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    if not kaggle_username or not kaggle_key:
        raise EnvironmentError(f"Kaggle API credentials are missing. Check {kaggle_env_path}.")

    try:
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                config["kaggle"]["competition"],
                "-p",
                str(raw_dir),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Kaggle download failed: {e}")

    # Extract dataset
    zip_path = raw_dir / f"{config['kaggle']['competition']}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        zip_path.unlink()
    logger.info("Dataset downloaded and extracted successfully.")


def preprocess(text: str) -> str:
    """
    Preprocess text data specifically for BERT sequence classification.

    Arguments:
        text: string, raw tweet text.
        
    Returns:
        string, preprocessed tweet.
    """
    if not text or pd.isna(text):
        return ""

    # Lowercase the text
    text = text.lower()

    # Expand contractions (e.g., don't -> do not)
    text = contractions.fix(text)

    # Normalize Twitter handles (@username -> @user)
    text = re.sub(r"@\w+", "@", text)

    # Normalize URLs (e.g., http://example.com -> http)
    text = re.sub(r"http\S+|www\S+|https\S+", "http", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove extra special characters, keeping basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:']", "", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_data(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> None:
    """
    Preprocess raw data and save it to the processed directory.

    Arguments:
        raw_dir: Path, the directory containing raw data files.
        processed_dir: Path, the directory to save processed data files.
    """
    logger.info("Preprocessing data...")
    processed_dir.mkdir(parents=True, exist_ok=True)

    for file_name in config["data"]["files"]:
        data = pd.read_csv(raw_dir / file_name)
        data["text"] = data["text"].fillna("").apply(preprocess)
        output_file = processed_dir / f"{file_name.split('.')[0]}_processed.csv"
        data.to_csv(output_file, index=False)
        logger.info(f"Processed data saved to {output_file}")
    logger.info("Preprocessing complete.")


def main() -> None:
    """
    Run the data pipeline by downloading and preprocessing the data.
    """
    download_data()
    preprocess_data()



if __name__ == "__main__":
    logger.info("Starting data pipeline...")
    main()
    logger.info("Data pipeline complete.")
