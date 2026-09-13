"""Thin wrapper around the backend API.

The base URL always comes from the API_BASE_URL environment variable - it is never
hard-coded, so the same frontend can point at a local, container, or remote backend.
"""
from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

# Generation on a local 7B model can take several seconds; a short timeout would make a
# working backend look broken.
HEALTH_TIMEOUT = 10
QUERY_TIMEOUT = 180


class APIError(Exception):
    """Raised with a message that is safe to show a user."""


def _friendly(exc: Exception) -> APIError:
    if isinstance(exc, requests.exceptions.ConnectionError):
        return APIError(
            f"Cannot reach the backend at {API_BASE_URL}. "
            "Start it with:  cd backend && uvicorn app.main:app --reload"
        )
    if isinstance(exc, requests.exceptions.Timeout):
        return APIError("The backend took too long to respond. Is the model still loading?")
    if isinstance(exc, requests.exceptions.HTTPError):
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            detail = exc.response.text[:200] if exc.response is not None else ""
        return APIError(detail or f"Backend returned {exc.response.status_code}.")
    return APIError(f"Unexpected error talking to the backend: {exc}")


def health() -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=HEALTH_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        raise _friendly(exc) from exc


def ask(question: str, state: str | None = None, top_k: int = 5) -> dict:
    payload = {"question": question, "top_k": top_k}
    if state:
        payload["state"] = state
    try:
        r = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=QUERY_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        raise _friendly(exc) from exc


def ask_image(image_bytes: bytes, filename: str, mime: str,
              state: str | None = None, question: str | None = None) -> dict:
    data = {}
    if state:
        data["state"] = state
    if question:
        data["question"] = question
    try:
        r = requests.post(
            f"{API_BASE_URL}/query/image",
            files={"file": (filename, image_bytes, mime)},
            data=data,
            timeout=QUERY_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        raise _friendly(exc) from exc
