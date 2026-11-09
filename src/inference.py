"""
Prediction code shared by the web app and the tests.

WHY a separate module: the app (app.py) is only UI. Keeping the model loading
and preprocessing here means the real-world test, the non-corn test and the
app all run exactly the same code, so a number in the README is a number the
app would actually show.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image

from src.data import fill_background

ROOT = Path(__file__).resolve().parent.parent  # relative paths: works on any machine
IMG_SIZE = 224
LOW_CONFIDENCE = 0.6
DISPLAY = {
    "Corn_(maize)___Common_rust_": "Common rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Northern Leaf Blight",
    "Corn_(maize)___healthy": "Healthy",
}


def load():
    """
    Load the model and its class order.

    WHY read class_names.json: index i of the model output means
    class_names[i], in the order training used. Hard-coding a list here risks
    a silent mismatch that mislabels every prediction.
    """
    model = tf.keras.models.load_model(ROOT / "models" / "corn_disease_model.keras", compile=False)
    with open(ROOT / "models" / "class_names.json") as f:
        class_names = json.load(f)
    return model, class_names


def preprocess(img: Image.Image) -> np.ndarray:
    """
    Must match what the model saw at test time, step for step:
    1. convert("RGB") - a PNG with transparency has 4 channels (RGBA), a
       greyscale photo has 1; the model needs exactly 3 or it crashes.
    2. resize to 224x224, bilinear - same as image_dataset_from_directory.
    3. add a batch dimension - models take a batch of images: (1, 224, 224, 3).
    4. replace a black background with grey - the same fix applied to the
       test set (only triggers when >5% of the image is pure black).

    NOT here: mobilenet_v2.preprocess_input. The model has that scaling built
    in as its first layer, so it expects raw 0-255 pixels. Applying it here
    too would scale the pixels twice and quietly break every prediction.
    """
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)[None, ...]
    return fill_background(tf.constant(x), training=False).numpy()


def predict(model, class_names, img: Image.Image) -> dict:
    """
    Return {display_name: probability} for ALL three classes, highest first.

    WHY all three: a 51/49 split and a 99/1 split are very different answers,
    but would show the same single winning label.
    """
    probs = model.predict(preprocess(img), verbose=0)[0]
    result = {DISPLAY[c]: float(p) for c, p in zip(class_names, probs)}
    return dict(sorted(result.items(), key=lambda kv: -kv[1]))
