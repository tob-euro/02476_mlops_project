import yaml
from pathlib import Path
import logging


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from a YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def setup_logger(log_file: str, level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
