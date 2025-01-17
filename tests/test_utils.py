import pytest
from twitter_classification.utils import load_config


def test_load_config():
    config = load_config("configs/config.yaml")
    assert "model" in config
    assert "training" in config
