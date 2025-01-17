import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from loguru import logger
from datetime import datetime

from twitter_classification.utils import load_config
from twitter_classification.data import TextDataset
from twitter_classification.model import build_model

# Load configuration
config = load_config()

def evaluate_model(
    model_dir: str = config["paths"]["output_dir"],
    eval_dataset_path: str = config["evaluation"]["dataset"],
    sample_submission_path: str = config["paths"]["sample_submission"],
    batch_size: int = config["evaluation"]["batch_size"],
    max_seq_length: int = config["evaluation"]["max_seq_length"],
    results_dir: str = config["paths"]["results_dir"],
    model_name: str = config["model"]["name"],
    device: torch.device | None = None,
) -> None:
    """
    Evaluates a trained model and generates a Kaggle submission file.

    Arguments:
        model_dir: string, path to the trained model directory.
        eval_dataset_path: string, path to the evaluation dataset.
        sample_submission_path: string, path to the sample submission template.
        batch_size: integer, batch size for evaluation.
        max_seq_length: integer, maximum sequence length for tokenization.
        results_dir: string, directory to save evaluation results.
        model_name: string, the Hugging Face model name.
        device: torch.device or None, the device to evaluate the model on (CPU or CUDA).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load evaluation dataset
    logger.info("Loading evaluation dataset...")
    dataset = TextDataset(
        data_dir=Path(eval_dataset_path).parent,
        file_name=Path(eval_dataset_path).name,
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load tokenizer and model
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = build_model(model_name=model_name, num_labels=config["model"]["num_labels"], model_dir=model_dir)
    model.to(device)

    # Evaluate
    logger.info("Evaluating model...")
    model.eval()
    all_predictions: list[int] = []

    with torch.no_grad():
        for batch in data_loader:
            texts = batch["text"]
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )
            inputs = {key: val.to(device) for key, val in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())

    # Generate submission file
    logger.info("Generating Kaggle submission...")
    sample_submission = pd.read_csv(sample_submission_path)
    sample_submission["target"] = all_predictions

    # Ensure results directory exists
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Save submission file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_file = results_path / f"submission_{timestamp}.csv"
    sample_submission.to_csv(submission_file, index=False)
    logger.info(f"Submission file saved to {submission_file}")

if __name__ == "__main__":
    evaluate_model()
