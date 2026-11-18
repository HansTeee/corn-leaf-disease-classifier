"""
Streamlit web app for the corn leaf disease classifier.

    streamlit run app.py      -> opens http://localhost:8501

WHY Streamlit (not Gradio): Hugging Face Spaces now charges for hosting Gradio
apps, while Streamlit Community Cloud hosts Streamlit apps for free, straight
from the GitHub repo.
"""
import io
import os
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

from src import inference

ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT / "docs" / "examples"

# WHY optional: in the Docker image the model runs in the
# FastAPI service and this UI calls it over HTTP (API_URL is set there). On
# Streamlit Community Cloud there is no separate API process, so the app loads
# the model itself. Both paths use src/inference.py, so the numbers are identical.
API_URL = os.environ.get("API_URL", "").rstrip("/")

st.set_page_config(page_title="Corn Leaf Disease Classifier", page_icon="🌽", layout="centered")


# ---------------------------------------------------------------------------
# Load ONCE.
# WHY @st.cache_resource: Streamlit re-runs this whole script top to bottom on
# every click and upload. Without caching, it would reload the 10 MB model each
# time. With it, the model is loaded on the first visit and then kept in memory
# for every later visitor.
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    return inference.load()


st.title("🌽 Corn Leaf Disease Classifier")
st.markdown(
    "**Trained on PlantVillage lab images. Corn only, three classes: "
    "Common rust, Northern Leaf Blight, Healthy.**"
)
st.caption(
    "The training photos are single detached leaves on plain backgrounds under even lighting. "
    "On real field photos the model is much less accurate: it got 5 of 10 right in testing, and "
    "it usually mistakes real-world common rust for blight. It always picks one of the three "
    "classes and cannot recognise other diseases or images that aren't corn leaves."
)

# --- input: upload, or pick one of the example test images -----------------
examples = sorted(EXAMPLES_DIR.glob("*.jpg"))
uploaded = st.file_uploader("Upload a corn leaf photo", type=["jpg", "jpeg", "png", "webp"])
choice = st.selectbox(
    "…or try an example from the PlantVillage test set",
    ["(none)"] + [p.stem for p in examples],
)

image_bytes = None
if uploaded is not None:
    image_bytes = uploaded.getvalue()
elif choice != "(none)":
    image_bytes = (EXAMPLES_DIR / f"{choice}.jpg").read_bytes()
image = Image.open(io.BytesIO(image_bytes)) if image_bytes is not None else None


def predict_probs(data: bytes, img: Image.Image) -> dict:
    """{display_name: probability}, highest first - from the API if configured, else locally."""
    if API_URL:
        resp = requests.post(f"{API_URL}/predict", files={"file": ("image", data)}, timeout=60)
        resp.raise_for_status()
        return resp.json()["probabilities"]
    model, class_names = get_model()
    return inference.predict(model, class_names, img)


# --- prediction -------------------------------------------------------------
if image is not None:
    try:
        probs = predict_probs(image_bytes, image)
    except requests.RequestException as e:
        st.error(f"The prediction service is unavailable ({e.__class__.__name__}). Try again in a moment.")
        st.stop()
    top_label, top_p = next(iter(probs.items()))

    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption="Input", width="stretch")
    with right:
        st.subheader(top_label)
        # All three classes, highest first, each with its own bar.
        for label, p in probs.items():
            st.write(f"{label}: **{p:.1%}**")
            st.progress(p)

    if top_p < inference.LOW_CONFIDENCE:
        # WHY: softmax ALWAYS produces three numbers that sum to 1, even for a
        # photo of a cat. The model cannot say "I don't know", so the UI must.
        st.warning(
            f"**Uncertain ({top_p:.0%} top confidence).** The model isn't sure. This may not be "
            "a single corn leaf, or it may be a disease the model was never trained on. "
            "Don't rely on this result.",
            icon="⚠️",
        )
    else:
        st.info(
            f"Top prediction confidence: {top_p:.0%}. High confidence does **not** mean correct: "
            "in real-world testing, every wrong answer had over 60% confidence."
        )

st.divider()
st.caption(
    "This is a learning demo, **not** a substitute for advice from an agronomist or your local "
    "agricultural office. Don't make crop treatment decisions based on it. "
    "[Source code and full evaluation](https://github.com/HansTeee/corn-leaf-disease-classifier)"
)
