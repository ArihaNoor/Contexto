"""Unit tests for the ingestion, memory, provider, and config layers."""

import json

import httpx
import pytest

from app.config import settings
from app.services import llm, memory
from app.services.ingestion import ingest_pdf
from app.services.vectorstore import collection_name, session_exists
from tests.conftest import build_pdf


# --------------------------------------------------------------------- ingest


def test_ingest_pdf_returns_session_and_chunk_count(sample_pdf):
    session_id, total_chunks = ingest_pdf(sample_pdf, "sample.pdf")
    assert total_chunks >= 1
    assert session_exists(session_id)


def test_ingest_pdf_stores_one_based_pages_and_source(sample_pdf):
    from app.services.vectorstore import get_vectorstore

    session_id, _ = ingest_pdf(sample_pdf, "sample.pdf")
    records = get_vectorstore(session_id).get()

    assert records["metadatas"], "expected indexed chunks"
    for metadata in records["metadatas"]:
        assert metadata["source"] == "sample.pdf"
        assert metadata["page"] >= 1


def test_ingest_pdf_raises_on_textless_pdf(blank_pdf):
    with pytest.raises(ValueError, match="No extractable text"):
        ingest_pdf(blank_pdf, "scan.pdf")


def test_ingest_splits_long_documents_into_multiple_chunks():
    long_page = "Retrieval augmented generation. " * 200
    pdf = build_pdf([long_page, long_page])

    _, total_chunks = ingest_pdf(pdf, "long.pdf")
    assert total_chunks > 2, "long documents should split past one chunk per page"


def test_each_ingest_gets_an_isolated_collection(sample_pdf):
    first, _ = ingest_pdf(sample_pdf, "a.pdf")
    second, _ = ingest_pdf(sample_pdf, "b.pdf")

    assert first != second
    assert collection_name(first) != collection_name(second)
    assert session_exists(first) and session_exists(second)


def test_session_exists_is_false_for_unknown_id():
    assert session_exists("never-created") is False


# --------------------------------------------------------------------- memory


def test_memory_is_isolated_per_session():
    memory.append_turn("a", "question a", "answer a")
    memory.append_turn("b", "question b", "answer b")

    assert [m.content for m in memory.get_history("a").messages] == [
        "question a",
        "answer a",
    ]
    assert len(memory.get_history("b").messages) == 2


def test_memory_trims_to_the_last_n_turns():
    for i in range(memory.MAX_TURNS + 5):
        memory.append_turn("s", f"q{i}", f"a{i}")

    messages = memory.get_history("s").messages
    assert len(messages) == memory.MAX_TURNS * 2
    assert messages[0].content == "q5", "oldest turns should be evicted first"
    assert messages[-1].content == f"a{memory.MAX_TURNS + 4}"


def test_clear_history_is_safe_for_unknown_session():
    memory.clear_history("never-existed")  # must not raise


# ------------------------------------------------------------------ provider


MESSAGES = [{"role": "user", "content": "hi"}]


def mock_provider(monkeypatch, handler):
    """Point the LLM client at an in-process fake provider."""
    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://provider.test/v1"
    )
    monkeypatch.setattr(llm, "get_client", lambda: client)
    return client


def chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"content": content}}]}
    )


def test_generate_returns_the_assistant_message(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return chat_response("  A grounded answer.  ")

    mock_provider(monkeypatch, handler)

    assert llm.generate(MESSAGES) == "A grounded answer."
    assert captured["url"].endswith("/chat/completions")
    assert captured["body"]["model"] == settings.resolved_model
    assert captured["body"]["stream"] is False


def test_generate_sends_the_bearer_token(monkeypatch):
    monkeypatch.setattr(settings, "llm_api_key", "secret-key")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return chat_response("ok")

    mock_provider(monkeypatch, handler)
    llm.generate(MESSAGES)

    assert seen["auth"] == "Bearer secret-key"


def test_missing_api_key_raises_actionable_error(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "huggingface")
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "huggingfacehub_api_token", "")

    with pytest.raises(llm.LLMError, match="HUGGINGFACEHUB_API_TOKEN"):
        llm.generate(MESSAGES)


def test_local_provider_needs_no_api_key(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "llm_api_key", "")

    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return chat_response("local answer")

    mock_provider(monkeypatch, handler)
    assert llm.generate(MESSAGES) == "local answer"


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, {}, "credentials"),
        (402, {}, "out of inference credits"),
        (429, {}, "Rate limited"),
        (400, {"error": {"code": "model_not_supported"}}, "LLM_MODEL"),
        (404, {}, "LLM_MODEL"),
        (500, {}, "HTTP 500"),
    ],
)
def test_provider_errors_are_translated(monkeypatch, status, body, expected):
    mock_provider(monkeypatch, lambda request: httpx.Response(status, json=body))

    with pytest.raises(llm.LLMError, match=expected):
        llm.generate(MESSAGES)


def test_unreachable_local_server_suggests_ollama(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "ollama")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    mock_provider(monkeypatch, handler)

    with pytest.raises(llm.LLMError, match="ollama serve"):
        llm.generate(MESSAGES)


def test_empty_completion_is_rejected(monkeypatch):
    mock_provider(monkeypatch, lambda request: chat_response(""))

    with pytest.raises(llm.LLMError, match="empty response"):
        llm.generate(MESSAGES)


def test_malformed_response_is_rejected(monkeypatch):
    mock_provider(
        monkeypatch, lambda request: httpx.Response(200, json={"unexpected": True})
    )

    with pytest.raises(llm.LLMError, match="malformed"):
        llm.generate(MESSAGES)


def sse(*chunks: str) -> bytes:
    lines = []
    for chunk in chunks:
        payload = json.dumps({"choices": [{"delta": {"content": chunk}}]})
        lines.append(f"data: {payload}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def test_stream_yields_tokens_in_order(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200, content=sse("Hello", " ", "world"),
            headers={"Content-Type": "text/event-stream"},
        )

    mock_provider(monkeypatch, handler)

    assert list(llm.stream(MESSAGES)) == ["Hello", " ", "world"]


def test_stream_skips_keepalives_and_malformed_frames(monkeypatch):
    content = (
        b": keep-alive\n\n"
        b"data: not-json\n\n"
        b'data: {"choices": [{"delta": {}}]}\n\n'
        b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    mock_provider(monkeypatch, lambda request: httpx.Response(200, content=content))

    assert list(llm.stream(MESSAGES)) == ["ok"]


def test_stream_translates_provider_errors(monkeypatch):
    mock_provider(monkeypatch, lambda request: httpx.Response(402, json={}))

    with pytest.raises(llm.LLMError, match="out of inference credits"):
        list(llm.stream(MESSAGES))


def test_stream_rejects_a_token_less_response(monkeypatch):
    mock_provider(
        monkeypatch, lambda request: httpx.Response(200, content=b"data: [DONE]\n\n")
    )

    with pytest.raises(llm.LLMError, match="empty response"):
        list(llm.stream(MESSAGES))


# -------------------------------------------------------------------- config


def test_provider_presets_resolve_base_url_and_model(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "llm_base_url", "")
    monkeypatch.setattr(settings, "llm_model", "")

    assert settings.resolved_base_url == "http://localhost:11434/v1"
    assert settings.resolved_model == "llama3.2"
    assert settings.requires_api_key is False


def test_explicit_overrides_beat_the_preset(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "llm_base_url", "https://my-vllm.internal/v1/")
    monkeypatch.setattr(settings, "llm_model", "my-finetune")

    assert settings.resolved_base_url == "https://my-vllm.internal/v1"
    assert settings.resolved_model == "my-finetune"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "not-a-provider")
    monkeypatch.setattr(settings, "llm_base_url", "")

    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        _ = settings.resolved_base_url
