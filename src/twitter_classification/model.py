from pathlib import Path
from transformers import AutoModelForSequenceClassification
from twitter_classification.utils import load_config

# Load configuration
config = load_config()

def build_model(
    model_name: str = config["model"]["name"], 
    num_labels: int = config["model"]["num_labels"], 
    model_dir: str | None = None
) -> AutoModelForSequenceClassification:
    """
    Builds a BERT model for sequence classification. Optionally loads pre-trained weights.

    Arguments:
        model_name: string, the Hugging Face model name to use.
        num_labels: integer, the number of output labels for classification.
        model_dir: string or None, the directory containing pre-trained model weights.

    Returns:
        A Hugging Face Transformers model configured for sequence classification.
    """
    if model_dir and Path(model_dir).exists():
        # Load pre-trained model from the directory
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    else:
        # Initialize a new model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    return model
