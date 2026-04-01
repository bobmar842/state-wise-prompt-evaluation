"""
LLM service — provider-agnostic wrapper around LiteLLM.
"""
import os
import time
from typing import Optional

from .config import OPENAI_API_KEY

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def call_llm(
    messages: list[dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> dict:
    """
    Call an LLM and return standardised result.

    Returns:
        {"text": str, "model": str, "usage": dict, "processing_time": float}
    """
    try:
        import litellm
    except ImportError:
        raise RuntimeError("litellm not installed. Run: pip install litellm")

    litellm.drop_params = True
    start = time.time()

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if OPENAI_API_KEY:
        kwargs["api_key"] = OPENAI_API_KEY
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    response = litellm.completion(**kwargs)
    text = response.choices[0].message.content if response.choices else ""
    usage = getattr(response, "usage", None)

    return {
        "text": text or "",
        "model": model,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
        },
        "processing_time": time.time() - start,
    }
