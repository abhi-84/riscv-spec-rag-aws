"""Retrieval: query rewrite, vector search, merge, rerank."""
import logging
from functools import lru_cache

from chromadb import PersistentClient
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry

from app.config import get_settings
from app.rag.llm import chat, retry_policy

logger = logging.getLogger(__name__)


class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


# ---- clients are created lazily (not at import): faster cold starts, easier tests
@lru_cache
def _openai() -> OpenAI:
    return OpenAI()


@lru_cache
def _collection():
    s = get_settings()
    return PersistentClient(path=str(s.db_path)).get_or_create_collection(s.collection_name)


# ---- query rewrite
@retry(**retry_policy)
def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    """Rewrite the user's question into a short, specific knowledge-base query."""
    history = history or []
    message = f"""
You are in a conversation with an engineer, answering questions about RISC-V specifications.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a short, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    response = chat([{"role": "system", "content": message}])
    return response.choices[0].message.content.strip()


# ---- vector search
def fetch_context_unranked(question: str) -> list[Result]:
    s = get_settings()
    query = _openai().embeddings.create(model=s.embedding_model, input=[question]).data[0].embedding
    results = _collection().query(query_embeddings=[query], n_results=s.retrieval_k)
    return [
        Result(page_content=doc, metadata=meta)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def merge_chunks(chunks: list[Result], extra: list[Result]) -> list[Result]:
    """Append chunks from `extra` that are not already present (by text)."""
    merged = chunks[:]
    existing = {c.page_content for c in chunks}
    for chunk in extra:
        if chunk.page_content not in existing:
            merged.append(chunk)
            existing.add(chunk.page_content)
    return merged


# ---- rerank
def apply_order(order: list[int], chunks: list) -> list:
    """Pure function: turn LLM-provided 1-based ids into a chunk list.
    Drops invalid/duplicate ids and appends any chunks the LLM skipped."""
    seen = set()
    result = []
    for i in order:
        if 1 <= i <= len(chunks) and i not in seen:
            seen.add(i)
            result.append(chunks[i - 1])
    for idx, chunk in enumerate(chunks, start=1):
        if idx not in seen:
            result.append(chunk)
    return result


@retry(**retry_policy)
def _rerank_llm(question: str, chunks: list[Result]) -> list[int]:
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = (
        f"The user has asked the following question:\n\n{question}\n\n"
        "Order all the chunks of text by relevance to the question, from most relevant to least relevant. "
        "Include all the chunk ids you are provided with, reranked.\n\nHere are the chunks:\n\n"
    )
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    response = chat(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        response_format=RankOrder,
    )
    return RankOrder.model_validate_json(response.choices[0].message.content).order


def rerank(question: str, chunks: list[Result]) -> list[Result]:
    """Optional step: on failure, keep retrieval order instead of failing the request."""
    try:
        order = _rerank_llm(question, chunks)
    except Exception:
        logger.warning("Rerank failed; falling back to retrieval order", exc_info=True)
        return chunks
    return apply_order(order, chunks)


# ---- full retrieval pipeline
def fetch_context(question: str, history: list[dict] | None = None) -> list[Result]:
    rewritten = rewrite_query(question, history)
    chunks = merge_chunks(fetch_context_unranked(question), fetch_context_unranked(rewritten))
    return rerank(question, chunks)[: get_settings().final_k]
