"""Vector-store retrieval.

The store is built by notebooks/rag_pipeline.ipynb and loaded once at startup; nothing
here re-chunks or re-embeds the corpus at request time.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    state: str
    page: int
    source_file: str
    distance: float


class Retriever:
    """Loaded once in the FastAPI lifespan and reused for every request."""

    def __init__(self, store_path: Path | None = None) -> None:
        store_path = Path(store_path or settings.vector_store_path)
        if not store_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {store_path}. "
                "Run notebooks/rag_pipeline.ipynb (section 2.7) to build it."
            )

        # config.json records what the store was actually built with, so the query
        # embedder can never silently drift from the document embedder.
        config_file = store_path / "config.json"
        self.config: dict = json.loads(config_file.read_text()) if config_file.exists() else {}
        model_name = self.config.get("embedding_model", settings.embedding_model)
        collection_name = self.config.get("collection_name", settings.collection_name)

        # CPU is deliberate: one short question per request is a few milliseconds here,
        # and it leaves the whole GPU to the LLM.
        self.embedder = SentenceTransformer(model_name, device="cpu")
        self.collection = chromadb.PersistentClient(path=str(store_path)).get_collection(
            collection_name
        )
        self.embedding_model = model_name
        logger.info(
            "retriever ready: %d chunks, embedder=%s, collection=%s",
            self.collection.count(), model_name, collection_name,
        )

    @property
    def chunk_count(self) -> int:
        return self.collection.count()

    @property
    def states(self) -> list[str]:
        return self.config.get("states", [])

    def retrieve(self, question: str, state: str | None = None, k: int | None = None) -> list[Chunk]:
        k = k or settings.top_k
        vector = self.embedder.encode([question], normalize_embeddings=True).tolist()
        result = self.collection.query(
            query_embeddings=vector,
            n_results=k,
            where={"state": state} if state else None,
        )
        if not result["documents"] or not result["documents"][0]:
            return []
        return [
            Chunk(text=doc, state=meta["state"], page=meta["page"],
                  source_file=meta["source_file"], distance=dist)
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]


def format_sources(chunks: list[Chunk]) -> list[str]:
    """One citation per distinct (file, page), in retrieval-rank order."""
    seen: set[tuple[str, int]] = set()
    sources: list[str] = []
    for c in chunks:
        key = (c.source_file, c.page)
        if key not in seen:
            seen.add(key)
            sources.append(f"{c.source_file} - p.{c.page} ({c.state})")
    return sources


def cited_sources(chunks: list[Chunk], answer: str) -> list[str]:
    """Only the blocks the answer actually cited as [n].

    Returning every retrieved chunk would overstate the grounding: the model
    typically uses one or two of the five. Falls back to all chunks if the answer
    carries no parsable citation, so a source list is never silently empty.
    """
    indices = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    used = [chunks[i - 1] for i in indices if 1 <= i <= len(chunks)]
    return format_sources(used or chunks)
