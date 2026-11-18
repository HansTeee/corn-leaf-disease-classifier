"""
FastAPI inference service for the corn leaf disease classifier.

    uvicorn api.main:app --port 8000      -> http://localhost:8000/docs

WHY a separate API: the model sits behind one HTTP endpoint, so the Streamlit UI
(or anything else: a phone app, curl, a batch script) can use it without loading
TensorFlow itself. Prediction goes through src/inference.py, the same code the
evaluation used, so the API returns exactly the numbers reported in the README.
"""
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src import inference

MAX_BYTES = 10 * 1024 * 1024  # reject uploads over 10 MB
STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # WHY load at startup: loading the model takes seconds; doing it once means
    # every request after that only pays for the prediction itself.
    STATE["model"], STATE["class_names"] = inference.load()
    yield
    STATE.clear()


app = FastAPI(
    title="Corn Leaf Disease Classifier API",
    description="MobileNetV2 fine-tuned on PlantVillage corn leaves: Common rust, Northern Leaf Blight, Healthy.",
    version="1.0.0",
    lifespan=lifespan,
)


class Prediction(BaseModel):
    label: str
    confidence: float
    low_confidence: bool
    probabilities: dict[str, float]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in STATE, "classes": STATE.get("class_names", [])}


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)) -> Prediction:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File larger than 10 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File is not a readable image.") from None

    probs = inference.predict(STATE["model"], STATE["class_names"], img)
    label, conf = next(iter(probs.items()))
    return Prediction(
        label=label,
        confidence=conf,
        low_confidence=conf < inference.LOW_CONFIDENCE,
        probabilities=probs,
    )
