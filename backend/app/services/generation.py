"""Prompt construction and the Ollama call.

The prompt is the grounding mechanism: numbered context blocks, an instruction to use
nothing else, required inline [n] citations, and an exact refusal string when the
context does not contain the answer.
"""
from __future__ import annotations

import logging
import re

import ollama

from app.core.config import settings
from app.services.retrieval import Chunk

logger = logging.getLogger(__name__)

REFUSAL = "I could not find this in the handbook."

SYSTEM_PROMPT = (
    "You are a driving-rules assistant. You answer ONLY from the numbered context blocks "
    "supplied by the user. Those blocks are excerpts from official US state driver handbooks.\n\n"
    "Rules:\n"
    "1. Use ONLY facts stated in the context. Never use outside knowledge.\n"
    "2. Cite the block you used inline, like [1] or [2], immediately after the claim it supports.\n"
    f"3. If the context does not contain the answer, reply with exactly: {REFUSAL}\n"
    "4. If the question is about a specific state, prefer blocks from that state's handbook.\n"
    "5. Be concise - two or three sentences unless the rule is a list."
)


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    blocks = "\n\n".join(
        f"[{i}] ({c.state} handbook, p.{c.page})\n{c.text}" for i, c in enumerate(chunks, 1)
    )
    return f"CONTEXT:\n{blocks}\n\nQUESTION: {question}"


def is_grounded(answer: str) -> bool:
    """Grounded means the model either cited a block or explicitly refused."""
    if REFUSAL.lower() in answer.lower():
        return True
    return bool(re.search(r"\[\d+\]", answer))


class Generator:
    """Holds the Ollama client; created once in the FastAPI lifespan."""

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self.model = model or settings.llm_model
        self.client = ollama.Client(host=host or settings.ollama_host)

    def ping(self) -> bool:
        try:
            names = {m.get("model", "") for m in self.client.list().get("models", [])}
            if self.model not in names:
                logger.warning(
                    "model %s not pulled; available: %s. Run: ollama pull %s",
                    self.model, sorted(names), self.model,
                )
                return False
            return True
        except Exception as exc:  # Ollama not running is the common case
            logger.warning("Ollama unreachable at %s: %s", settings.ollama_host, exc)
            return False

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return REFUSAL
        response = self.client.chat(
            model=self.model,
            options={
                "temperature": settings.llm_temperature,
                "num_predict": settings.llm_num_predict,
            },
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, chunks)},
            ],
        )
        return response["message"]["content"].strip()
