import os
from pathlib import Path

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = sorted((ROOT / "docs" / "examples").glob("*.jpg"))
PREFIX_TO_LABEL = {"rust": "Common rust", "blight": "Northern Leaf Blight", "healthy": "Healthy"}


def expected_label(path: Path) -> str:
    return PREFIX_TO_LABEL[path.stem.split("_")[0]]


@pytest.fixture(scope="session")
def model_and_classes():
    # WHY session scope: loading TensorFlow + the model takes seconds; do it once.
    from src import inference

    return inference.load()
