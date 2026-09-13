"""FastAPI application for the Road Rules RAG Assistant.

The vector store, embedding model, YOLO weights and Ollama client are all loaded once in
the lifespan handler and stashed on app.state - never per request.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import query as query_routes
from app.core.config import settings
from app.services.generation import Generator
from app.services.retrieval import Retriever
from app.services.vision import SignDetector
from app.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("starting %s", settings.app_name)

    app.state.retriever = Retriever()
    app.state.generator = Generator()
    app.state.detector = SignDetector()
    app.state.llm_reachable = app.state.generator.ping()

    logger.info(
        "ready: %d chunks | states=%s | llm=%s (reachable=%s) | yolo=%s",
        app.state.retriever.chunk_count,
        ", ".join(app.state.retriever.states) or "-",
        app.state.generator.model,
        app.state.llm_reachable,
        app.state.detector.available,
    )
    if app.state.detector.available:
        logger.info("sign classes: %s", ", ".join(app.state.detector.covered_classes))
    if not app.state.llm_reachable:
        logger.warning("Ollama is not reachable - /query will return 503 until it is running.")

    yield
    logger.info("shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=(
            "Ask a driving-rules question and get an answer grounded in official US state "
            "driver handbooks, with page citations. Upload a road-sign photo to look up the "
            "matching rule."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(query_routes.router)
    return app


app = create_app()
