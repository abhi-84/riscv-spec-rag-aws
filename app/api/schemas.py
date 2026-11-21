"""API contract: request and response models for /v1/chat and /v1/retrieve.

Pydantic validates every incoming request against these models BEFORE any LLM call,
and FastAPI turns them into the OpenAPI docs at /docs.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_QUESTION_CHARS = 2000
MAX_MESSAGE_CHARS = 4000
MAX_HISTORY_MESSAGES = 20
SNIPPET_CHARS = 300


class Message(BaseModel):
    """One turn of the conversation, sent back by the client (the API is stateless)."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={"examples": [{
            "question": "Is the V extension mandatory in RVA23?",
            "history": [
                {"role": "user", "content": "What is RVA23?"},
                {"role": "assistant", "content": "RVA23 is a RISC-V application profile..."},
            ],
        }]},
    )

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[Message] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)


class Source(BaseModel):
    """Where a piece of the answer came from."""
    source: str
    pages: str | None = None
    snippet: str = Field(description=f"First {SNIPPET_CHARS} characters of the chunk")

    @classmethod
    def from_chunk(cls, chunk) -> "Source":
        """Build a Source from a retrieval Result (page_content + metadata)."""
        meta = chunk.metadata or {}
        return cls(
            source=str(meta.get("source", "unknown")),
            pages=meta.get("pages"),
            snippet=chunk.page_content[:SNIPPET_CHARS],
        )


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    request_id: str


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrieveResponse(BaseModel):
    results: list[Source]
    request_id: str


class ErrorResponse(BaseModel):
    """One error format for every failure, so callers can handle errors uniformly."""
    error: str
    detail: str | None = None
    request_id: str | None = None
