"""API routes: health, text query, and image query."""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.schemas.query import (
    Citation, Detection, HealthResponse, QueryRequest, QueryResponse,
)
from app.services.generation import REFUSAL, is_grounded
from app.services.retrieval import best_excerpt, cited_chunks, format_sources

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def build_citations(chunks, answer, retriever) -> tuple[list[str], list[Citation]]:
    """Resolve the answer's [n] markers into plain sources and quoted citations."""
    used = cited_chunks(chunks, answer)
    citations = [
        Citation(
            source_file=c.source_file,
            page=c.page,
            state=c.state,
            handbook=retriever.handbook_title(c.state),
            snippet=best_excerpt(c.text, answer),
            distance=round(c.distance, 4),
        )
        for c in used
    ]
    return format_sources(used), citations


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    state = request.app.state
    return HealthResponse(
        status="ok",
        chunks=state.retriever.chunk_count,
        embedding_model=state.retriever.embedding_model,
        llm_model=state.generator.model,
        llm_reachable=state.llm_reachable,
        yolo_loaded=state.detector.available,
        states=state.retriever.states,
    )


@router.post("/query", response_model=QueryResponse, tags=["query"])
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    state = request.app.state
    chunks = state.retriever.retrieve(payload.question, payload.state, payload.top_k)
    if not chunks:
        return QueryResponse(answer=REFUSAL, sources=[])

    try:
        answer = state.generator.generate(payload.question, chunks)
    except Exception as exc:
        logger.exception("generation failed")
        raise HTTPException(
            status_code=503,
            detail=f"The language model is unavailable. Is Ollama running? ({exc})",
        ) from exc

    refused = REFUSAL.lower() in answer.lower()
    logger.info(
        "q=%r state=%s chunks=%d grounded=%s",
        payload.question[:60], payload.state, len(chunks), is_grounded(answer),
    )
    if refused:
        return QueryResponse(answer=answer, sources=[], citations=[])
    sources, citations = build_citations(chunks, answer, state.retriever)
    return QueryResponse(answer=answer, sources=sources, citations=citations)


@router.post("/query/image", response_model=QueryResponse, tags=["query"])
async def query_image(
    request: Request,
    file: UploadFile = File(..., description="Photo containing a road sign."),
    state: str | None = Form(None),
    question: str | None = Form(None),
) -> QueryResponse:
    app_state = request.app.state
    if not app_state.detector.available:
        raise HTTPException(
            status_code=503,
            detail="Vision model not loaded. Train it with notebooks/yolo_training_colab.ipynb.",
        )
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type {file.content_type!r}. "
                   f"Expected one of {sorted(ALLOWED_IMAGE_TYPES)}.",
        )

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    tmp_path = Path(tempfile.mkdtemp()) / f"upload{suffix}"
    try:
        with tmp_path.open("wb") as out:
            shutil.copyfileobj(file.file, out, length=1024 * 1024)
        if tmp_path.stat().st_size > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image larger than 10 MB.")

        detections = app_state.detector.detect(tmp_path)
        if not detections:
            return QueryResponse(
                answer="No road sign was detected in this image. "
                       "Try a closer or better-lit photo, or ask your question as text.",
                sources=[], detections=[],
            )

        derived, _top = app_state.detector.to_question(detections)
        # A typed question, when supplied, takes priority; the detection still adds context.
        final_question = f"{question.strip()} {derived}" if question else derived

        chunks = app_state.retriever.retrieve(final_question, state)
        try:
            answer = app_state.generator.generate(final_question, chunks)
        except Exception as exc:
            logger.exception("generation failed")
            raise HTTPException(status_code=503, detail=f"LLM unavailable: {exc}") from exc

        refused = REFUSAL.lower() in answer.lower()
        sources, citations = ([], []) if refused else build_citations(
            chunks, answer, app_state.retriever
        )
        return QueryResponse(
            answer=answer,
            sources=sources,
            citations=citations,
            detections=[Detection(**d) for d in detections],
        )
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)
