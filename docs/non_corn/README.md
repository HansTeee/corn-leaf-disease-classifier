# Non-corn inputs (Phase 3 test)

Images that are not corn leaves, sent to the running app on 2026-11-09 (first built with Gradio; the current Streamlit app uses the same prediction code in `src/inference.py` and gives identical numbers)
with the model in `models/corn_disease_model.keras`. Both images ship with matplotlib's
sample data (`grace_hopper.jpg` is a public-domain US Navy photo).

| Image | Northern Leaf Blight | Common rust | Healthy | App message |
|---|---|---|---|---|
| `grace_hopper.jpg` (portrait of a person) | **52.19%** | 34.56% | 13.24% | ⚠️ Uncertain (52% top confidence) |
| `logo2.png` (matplotlib logo, RGBA) | **96.31%** | 2.30% | 1.39% | none: confidence is above the 0.6 threshold |

What this shows: the model can't say "this isn't a leaf". The confidence warning caught the
portrait but missed the logo, which got a 96% "Northern Leaf Blight" answer. Low confidence is
a useful hint, but high confidence doesn't mean the image is a corn leaf.
