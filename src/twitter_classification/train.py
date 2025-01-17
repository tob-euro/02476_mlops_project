import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from loguru import logger
from tqdm import tqdm

from twitter_classification.utils import load_config
from twitter_classification.model import build_model
from twitter_classification.data import TextDataset

# Load configuration
config = load_config()

def train_model(
    dataset_path: str = config["paths"]["dataset"],
    model_name: str = config["model"]["name"],
    output_dir: str = config["paths"]["output_dir"],
    num_labels: int = config["model"]["num_labels"],
    epochs: int = config["training"]["epochs"],
    batch_size: int = config["training"]["batch_size"],
    learning_rate: float = config["training"]["learning_rate"],
    max_seq_length: int = config["evaluation"]["max_seq_length"],
    device: Optional[torch.device] = None,
) -> None:
    """
    Trains a BERT model using PyTorch.

    Arguments:
        dataset_path: string, path to the processed dataset.
        model_name: string, Hugging Face model name.
        output_dir: string, directory to save the trained model.
        num_labels: integer, number of output labels.
        epochs: integer, number of training epochs.
        batch_size: integer, training batch size.
        learning_rate: float, learning rate for optimizer.
        max_seq_length: integer, maximum sequence length for tokenization.
        device: torch.device or None, device to train the model on (CPU or CUDA).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = TextDataset(
        data_dir=os.path.dirname(dataset_path),
        file_name=os.path.basename(dataset_path),
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load tokenizer and model
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = build_model(model_name=model_name, num_labels=num_labels)
    model.to(device)

    # Define optimizer and loss
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    logger.info("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress_bar:
            texts = batch["text"]
            labels = batch["label"].to(device)

            # Tokenize inputs
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )
            inputs = {key: val.to(device) for key, val in inputs.items()}

            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track accuracy and loss
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({"Loss": loss.item()})

        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save model
    logger.info(f"Saving model to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training complete.")

if __name__ == "__main__":
    train_model()
