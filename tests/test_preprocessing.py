"""Preprocessing bugs fail silently (wrong numbers, no error), so pin them down."""
import numpy as np
import tensorflow as tf
from PIL import Image

from src import inference
from src.data import fill_background


def test_mobilenet_scaling_is_inside_the_model(model_and_classes):
    """The app feeds raw 0-255 pixels; scaling to [-1, 1] must happen in the model."""
    model, _ = model_and_classes
    layer = model.get_layer("mobilenet_v2_preprocess")
    x = np.array([[[[0.0, 127.5, 255.0]]]], dtype=np.float32)
    np.testing.assert_allclose(layer(x).numpy().ravel(), [-1.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(
        layer(x).numpy(), tf.keras.applications.mobilenet_v2.preprocess_input(x.copy()), atol=1e-6
    )


def test_fill_background_replaces_black_background():
    x = np.full((1, 64, 64, 3), 120.0, dtype=np.float32)
    x[:, :, :32] = 0.0  # left half pure black, like a PlantVillage rust cut-out
    out = fill_background(tf.constant(x), training=False).numpy()
    assert out[0, 10, 5].mean() > 80  # black area now grey-ish
    np.testing.assert_allclose(out[0, :, 40:], 120.0)  # leaf pixels untouched


def test_fill_background_ignores_small_dark_patches():
    x = np.full((1, 64, 64, 3), 120.0, dtype=np.float32)
    x[:, :8, :8] = 0.0  # 1.6% of the image: a shadow, not a background
    out = fill_background(tf.constant(x), training=False).numpy()
    np.testing.assert_array_equal(out, x)


def test_fill_background_is_deterministic_at_eval_time():
    x = np.zeros((2, 32, 32, 3), dtype=np.float32)
    a = fill_background(tf.constant(x), training=False).numpy()
    b = fill_background(tf.constant(x), training=False).numpy()
    np.testing.assert_array_equal(a, b)


def test_preprocess_handles_any_image_mode():
    for mode in ("RGB", "RGBA", "L", "P"):
        img = Image.new(mode, (300, 180))
        x = inference.preprocess(img)
        assert x.shape == (1, 224, 224, 3)
        assert x.dtype == np.float32
