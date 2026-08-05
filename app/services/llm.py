"""Provider-agnostic LLM client.

Every provider Contexto supports — the Hugging Face router, OpenAI, Gemini's
compatibility endpoint, and local runtimes such as Ollama, vLLM or LM Studio —
exposes the same OpenAI ``/chat/completions`` protocol. So this module speaks
that protocol directly over httpx instead of binding to any one vendor SDK.
Swapping providers is a two-line environment change, not a code change.

This is the only module in the project that talks to a text-generation
provider; nothing else imports it beyond :mod:`app.services.rag`.
"""

import json
import logging
from collections.abc import Iterator
from functools import lru_cache

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the upstream provider cannot fulfil a generation request."""


@lru_cache(maxsize=1)
def get_client() -> httpx.Client:
    return httpx.Client(
        base_url=settings.resolved_base_url,
        timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=10.0),
    )


def _headers() -> dict[str, str]:
    api_key = settings.resolved_api_key
    if not api_key and settings.requires_api_key:
        raise LLMError(
            f"No API key configured for LLM_PROVIDER='{settings.llm_provider}'. "
            "Set LLM_API_KEY (or HUGGINGFACEHUB_API_TOKEN) in your .env — "
            "see .env.example."
        )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _payload(messages: list[dict], stream: bool) -> dict:
    return {
        "model": settings.resolved_model,
        "messages": messages,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "stream": stream,
    }


def _describe_http_error(response: httpx.Response) -> str:
    """Turn a provider error into something the user can act on."""
    body = response.text[:400]
    model, provider = settings.resolved_model, settings.llm_provider

    if response.status_code in (401, 403):
        return f"Provider rejected the credentials for '{provider}'. Check your API key."
    if response.status_code == 402:
        return (
            f"The '{provider}' account is out of inference credits. Add credits, or "
            "switch providers — e.g. LLM_PROVIDER=ollama for a free local model."
        )
    if response.status_code == 429:
        return f"Rate limited by '{provider}'. Wait a moment and try again."
    if "model_not_supported" in body or response.status_code == 404:
        return (
            f"The model '{model}' is not available from '{provider}'. "
            "Set LLM_MODEL to one this provider serves."
        )
    return f"The language model provider returned HTTP {response.status_code}."


def _request(messages: list[dict], stream: bool) -> httpx.Response:
    headers = _headers()
    request = get_client().build_request(
        "POST", "/chat/completions", json=_payload(messages, stream), headers=headers
    )
    try:
        return get_client().send(request, stream=stream)
    except httpx.ConnectError as exc:
        raise LLMError(
            f"Could not reach the language model at {settings.resolved_base_url}. "
            + (
                "Is the local model server running? (`ollama serve`)"
                if not settings.requires_api_key
                else "Check your network connection."
            )
        ) from exc
    except httpx.TimeoutException as exc:
        raise LLMError(
            f"The language model did not respond within "
            f"{settings.llm_timeout_seconds}s. Try a smaller model or raise "
            "LLM_TIMEOUT_SECONDS."
        ) from exc


def generate(messages: list[dict]) -> str:
    """Send a chat-completion request and return the assistant's reply text."""
    response = _request(messages, stream=False)
    try:
        if response.status_code >= 400:
            logger.error(
                "chat completion failed: %s %s",
                response.status_code,
                response.text[:400],
            )
            raise LLMError(_describe_http_error(response))

        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError("The language model returned a malformed response.") from exc
    finally:
        response.close()

    if not content or not content.strip():
        raise LLMError("The language model returned an empty response.")
    return content.strip()


def stream(messages: list[dict]) -> Iterator[str]:
    """Yield the assistant's reply incrementally as server-sent chunks arrive."""
    response = _request(messages, stream=True)
    try:
        if response.status_code >= 400:
            response.read()
            logger.error(
                "streaming chat completion failed: %s %s",
                response.status_code,
                response.text[:400],
            )
            raise LLMError(_describe_http_error(response))

        produced = False
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0]["delta"]
                token = delta.get("content")
            except (KeyError, IndexError, ValueError, TypeError, AttributeError):
                continue  # keep-alives and vendor-specific frames are not fatal
            if token:
                produced = True
                yield token

        if not produced:
            raise LLMError("The language model returned an empty response.")
    finally:
        response.close()
