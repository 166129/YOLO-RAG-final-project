"""Application settings, loaded from the environment / .env once at import time."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Road Rules RAG Assistant"

    # Artifacts produced by notebooks/rag_pipeline.ipynb (section 2.7).
    vector_store_path: Path = BACKEND_ROOT / "data" / "vector_store"
    yolo_weights_path: Path = BACKEND_ROOT / "models" / "signs_yolo.pt"

    # Overridden by config.json where the store records what it was built with.
    collection_name: str = "driver_handbooks"
    embedding_model: str = "all-MiniLM-L6-v2"

    llm_model: str = "qwen2.5:7b-instruct"
    ollama_host: str = "http://localhost:11434"
    llm_temperature: float = 0.0
    llm_num_predict: int = 400

    top_k: int = 5
    detection_confidence: float = 0.25

    # Comma-separated list of origins allowed to call this API.
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"

    log_level: str = "INFO"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
