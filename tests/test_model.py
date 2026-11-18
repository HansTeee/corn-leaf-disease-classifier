"""Model-level checks. If any of these fail, the deployed app would be wrong, so
CI refuses to deploy."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src import config, inference

from .conftest import EXAMPLES, ROOT, expected_label


def test_class_order_matches_training(model_and_classes):
    _, class_names = model_and_classes
    assert class_names == config.CLASS_NAMES


def test_output_is_a_probability_distribution(model_and_classes):
    model, _ = model_and_classes
    probs = model.predict(np.random.default_rng(0).uniform(0, 255, (4, 224, 224, 3)), verbose=0)
    assert probs.shape == (4, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs >= 0).all()


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_known_test_set_examples_are_classified_correctly(model_and_classes, path):
    model, class_names = model_and_classes
    probs = inference.predict(model, class_names, Image.open(path))
    label, conf = next(iter(probs.items()))
    assert label == expected_label(path)
    assert conf > 0.9


def test_reported_test_accuracy_meets_threshold():
    """Quality gate: the committed model must have been evaluated on the full held-out split."""
    m = json.loads((ROOT / "models" / "metrics.json").read_text())
    assert m["n_test"] == 490
    assert m["test_accuracy"] >= 0.95


def test_real_world_result_in_readme_is_still_true(model_and_classes):
    """README reports 5/10 on docs/real_world. If the model changes, this fails until the README is updated."""
    model, class_names = model_and_classes
    files = sorted(Path(ROOT / "docs" / "real_world").glob("*.jpg"))
    correct = sum(
        next(iter(inference.predict(model, class_names, Image.open(f)))) == expected_label(f) for f in files
    )
    assert (correct, len(files)) == (5, 10)
