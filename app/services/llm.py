"""LLM provider wrapper (Hugging Face Inference API, chat-completion protocol).

This module is the single seam between Contexto and its text-generation
provider. Anything that speaks the OpenAI-style ``messages`` protocol
(Hugging Face router, OpenAI, Gemini's compatibility endpoint, a local
vLLM/Ollama server) can be dropped in by reimplementing ``generate``
alone — no other module imports the provider SDK.
"""

import logging
from functools import lru_cache

from huggingface_hub import InferenceClient

from app.config import settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the upstream provider cannot fulfil a generation request."""


@lru_cache(maxsize=1)
def get_llm_client() -> InferenceClient:
    if not settings.huggingfacehub_api_token:
        raise LLMError(
            "HUGGINGFACEHUB_API_TOKEN is not set. Add it to your .env file "
            "(see .env.example) to enable answer generation."
        )
    return InferenceClient(token=settings.huggingfacehub_api_token)


def generate(messages: list[dict]) -> str:
    """Send a chat-completion request and return the assistant's reply text."""
    try:
        response = get_llm_client().chat_completion(
            messages=messages,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    except LLMError:
        raise
    except Exception as exc:
        logger.exception("Chat completion failed for model %s", settings.llm_model)
        if "model_not_supported" in str(exc):
            raise LLMError(
                f"The model '{settings.llm_model}' is not served by any inference "
                "provider enabled on your Hugging Face account. Pick another model "
                "via the LLM_MODEL environment variable."
            ) from exc
        raise LLMError(f"The language model provider is unavailable: {exc}") from exc

    content = response.choices[0].message.content
    if not content:
        raise LLMError("The language model returned an empty response.")
    return content.strip()
