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

    def handbook_title(self, state: str) -> str:
        """Human-readable document title, e.g. 'California Driver's Handbook (DL 600)'."""
        return self.config.get("sources", {}).get(state, f"{state} driver handbook")

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


def cited_chunks(chunks: list[Chunk], answer: str) -> list[Chunk]:
    """Only the blocks the answer actually cited as [n].

    Returning every retrieved chunk would overstate the grounding: the model
    typically uses one or two of the five. Falls back to all chunks if the answer
    carries no parsable citation, so a source list is never silently empty.
    """
    indices = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    used = [chunks[i - 1] for i in indices if 1 <= i <= len(chunks)]
    return used or chunks


def cited_sources(chunks: list[Chunk], answer: str) -> list[str]:
    return format_sources(cited_chunks(chunks, answer))


_WORD = re.compile(r"[a-z0-9.%]+")


def _content_words(text: str) -> set[str]:
    # Short tokens are mostly stopwords; numbers and percentages are kept because
    # they are usually the fact being cited ("0.01%", "2 seconds", "3 feet").
    return {
        w for w in _WORD.findall(text.lower())
        if len(w) > 3 or any(ch.isdigit() for ch in w)
    }


def best_excerpt(text: str, answer: str, max_chars: int = 300) -> str:
    """The window of `text` that best supports `answer`.

    A chunk is ~1000 characters; quoting all of it buries the relevant line. This
    scores each sentence by content-word overlap with the answer, then grows a
    window around the best one until it fills the budget.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return " ".join(text[:max_chars].split())

    wanted = _content_words(answer)
    best = max(range(len(sentences)), key=lambda i: len(_content_words(sentences[i]) & wanted))

    lo = hi = best
    out = sentences[best]
    while True:
        grew = False
        if hi + 1 < len(sentences) and len(out) + len(sentences[hi + 1]) + 1 <= max_chars:
            hi += 1
            out = f"{out} {sentences[hi]}"
            grew = True
        if lo - 1 >= 0 and len(out) + len(sentences[lo - 1]) + 1 <= max_chars:
            lo -= 1
            out = f"{sentences[lo]} {out}"
            grew = True
        if not grew:
            break

    out = " ".join(out.split())
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0]
    return f"{'…' if lo > 0 else ''}{out}{'…' if hi < len(sentences) - 1 else ''}"
