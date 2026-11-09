"""
Phase 3: Gradio web app for the corn leaf disease classifier.

    python app.py      -> open http://127.0.0.1:7860
"""
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

from src.data import fill_background

ROOT = Path(__file__).resolve().parent  # relative paths: works locally AND on HF Spaces
IMG_SIZE = 224
LOW_CONFIDENCE = 0.6
DISPLAY = {
    "Corn_(maize)___Common_rust_": "Common rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Northern Leaf Blight",
    "Corn_(maize)___healthy": "Healthy",
}

# ---------------------------------------------------------------------------
# Load ONCE, at startup.
# WHY: Gradio calls predict() for every image submitted. Loading a 10 MB model
# inside predict() would add seconds to every single request.
# ---------------------------------------------------------------------------
MODEL = tf.keras.models.load_model(ROOT / "models" / "corn_disease_model.keras", compile=False)
# WHY read the names from the file: index i of the model output means
# CLASS_NAMES[i], in the order training used. Hard-coding a list here risks a
# silent mismatch that mislabels every prediction.
with open(ROOT / "models" / "class_names.json") as f:
    CLASS_NAMES = json.load(f)


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


def predict(img):
    if img is None:
        return None, ""
    probs = MODEL.predict(preprocess(img), verbose=0)[0]
    # WHY return all three: a 51/49 split and a 99/1 split are very different
    # answers, but would show the same single winning label.
    result = {DISPLAY[c]: float(p) for c, p in zip(CLASS_NAMES, probs)}

    top = float(probs.max())
    if top < LOW_CONFIDENCE:
        # WHY: softmax ALWAYS produces three numbers that sum to 1 - even for a
        # photo of a cat. The model cannot say "I don't know", so the UI must.
        note = (f"⚠️ **Uncertain ({top:.0%} top confidence).** The model isn't sure. "
                "This may not be a single corn leaf, or it may be a disease the model "
                "was never trained on. Don't rely on this result.")
    else:
        note = (f"Top prediction confidence: {top:.0%}. High confidence does **not** mean "
                "correct: any image gets a confident answer if it looks enough like a "
                "training image.")
    return result, note


EXAMPLES = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs" / "examples").glob("*.*"))

DESCRIPTION = """
**Trained on PlantVillage lab images. Corn only, three classes: Common rust, Northern Leaf Blight, Healthy.**

The training photos are single detached leaves on plain backgrounds under even lighting.
Field photos (whole plants, mixed backgrounds, sunlight) are much harder, and accuracy on them is lower.
The model always picks one of the three classes. It cannot recognise other diseases or images that aren't corn leaves.
"""

ARTICLE = """
*This is a learning demo, **not** a substitute for advice from an agronomist or your local
agricultural office. Don't make crop treatment decisions based on it.*
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Corn leaf photo"),
    outputs=[gr.Label(num_top_classes=3, label="Prediction"), gr.Markdown()],
    title="🌽 Corn Leaf Disease Classifier",
    description=DESCRIPTION,
    article=ARTICLE,
    examples=EXAMPLES or None,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
