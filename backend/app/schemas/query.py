"""Request/response models for the query API."""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="A driving-rules question.",
        examples=["What is the BAC limit for drivers under 21?"],
    )
    state: str | None = Field(
        None,
        description="Restrict retrieval to one state's handbook. Omit to search all of them.",
        examples=["California"],
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve.")


class Detection(BaseModel):
    label: str = Field(..., description="Detected road-sign class.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: list[int] = Field(..., description="Pixel box as [x1, y1, x2, y2].")


class Citation(BaseModel):
    """A cited chunk, with the handbook text it was drawn from.

    `sources` above carries the same citations as plain strings (the shape the
    project spec requires); this is the richer form the UI renders, so a reader can
    see the sentence the answer came from without opening the PDF.
    """

    source_file: str
    page: int
    state: str
    handbook: str = Field(..., description="Human-readable document title.")
    snippet: str = Field(..., description="The passage of the chunk most relevant to the answer.")
    distance: float = Field(..., description="Cosine distance; lower is a closer match.")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Grounded answer with inline [n] citations.")
    sources: list[str] = Field(
        default_factory=list,
        description="Handbook and page for each chunk used. Empty when the assistant refused.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="The same citations with quoted handbook text. Empty when refused.",
    )
    detections: list[Detection] = Field(
        default_factory=list,
        description="Road signs found in an uploaded image. Empty for text queries.",
    )


class HealthResponse(BaseModel):
    status: str
    chunks: int
    embedding_model: str
    llm_model: str
    llm_reachable: bool
    yolo_loaded: bool
    states: list[str]
