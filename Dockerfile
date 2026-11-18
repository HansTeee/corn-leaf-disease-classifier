# Corn leaf disease classifier: FastAPI (model, port 8000) + Streamlit UI (port 8501).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    API_URL=http://127.0.0.1:8000 \
    HOME=/home/user

RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user:user src/ src/
COPY --chown=user:user api/ api/
COPY --chown=user:user models/corn_disease_model.keras models/class_names.json models/
COPY --chown=user:user docs/examples/ docs/examples/
COPY --chown=user:user app.py docker/start.sh ./
RUN chmod +x start.sh

USER user
EXPOSE 8501 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["./start.sh"]
