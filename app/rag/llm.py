"""Shared LLM helpers: one retry policy and one completion wrapper for the whole app."""
import logging

from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()

# Bounded retries for interactive requests: short waits, re-raise the original error.
retry_policy = dict(
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(_settings.retry_attempts),
    reraise=True,
)


def chat(messages: list[dict], **kwargs):
    """Single place where the app calls the chat model (timeout always applied)."""
    return completion(
        model=_settings.chat_model,
        messages=messages,
        timeout=_settings.llm_timeout_s,
        **kwargs,
    )
