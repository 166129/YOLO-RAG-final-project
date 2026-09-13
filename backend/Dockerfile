# Road Rules RAG Assistant - backend
#
# Build from the backend/ directory:
#     docker build -t road-rules-api .
#
# Ollama runs on the HOST, not in this container, so point the app at the host gateway:
#     docker run -p 8000:8000 \
#       -e OLLAMA_HOST=http://host.docker.internal:11434 \
#       --add-host=host.docker.internal:host-gateway \
#       road-rules-api
#
# On Docker Desktop (Windows/macOS) host.docker.internal resolves already and
# --add-host is unnecessary; on Linux it is required.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgl/libglib are needed by ultralytics' OpenCV dependency.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

# Requirements first so the dependency layer caches across code edits.
COPY requirements.txt .

# The CPU-only torch wheel keeps the image ~2 GB smaller; inference here is a single
# small YOLO model, and the LLM runs on the host under Ollama.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Artifacts built by notebooks/rag_pipeline.ipynb and yolo_training_colab.ipynb.
COPY data/vector_store ./data/vector_store
COPY models ./models
COPY app ./app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
