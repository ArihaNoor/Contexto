"""End-to-end API tests covering the ingest -> query -> clear lifecycle."""

import json

import pytest

from app.config import settings
from app.services import llm
from tests.conftest import FAKE_ANSWER


def ingest(client, pdf_bytes: bytes, filename: str = "sample.pdf"):
    return client.post(
        "/api/v1/context/ingest",
        files={"file": (filename, pdf_bytes, "application/pdf")},
    )


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"app": "Contexto", "status": "ok"}


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Contexto" in response.text


def test_ingest_indexes_chunks(client, sample_pdf):
    response = ingest(client, sample_pdf)
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["total_chunks"] >= 1
    assert len(body["session_id"]) == 32


def test_ingest_rejects_non_pdf(client):
    response = client.post(
        "/api/v1/context/ingest",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "pdf" in response.json()["detail"].lower()


def test_ingest_rejects_empty_file(client):
    response = ingest(client, b"", "empty.pdf")
    assert response.status_code == 400


def test_ingest_rejects_oversized_file(client):
    oversized = b"x" * (settings.max_file_size_bytes + 1)
    response = ingest(client, oversized, "huge.pdf")
    assert response.status_code == 413
    assert str(settings.max_file_size_mb) in response.json()["detail"]


def test_ingest_rejects_pdf_without_text(client, blank_pdf):
    response = ingest(client, blank_pdf, "scan.pdf")
    assert response.status_code == 422
    assert "no extractable text" in response.json()["detail"].lower()


def test_query_returns_grounded_answer_with_citations(client, sample_pdf, fake_llm):
    session_id = ingest(client, sample_pdf).json()["session_id"]

    response = client.post(
        "/api/v1/context/query",
        json={"session_id": session_id, "query": "What produces the embeddings?"},
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["answer"] == FAKE_ANSWER
    assert body["sources"], "expected at least one citation"
    for source in body["sources"]:
        assert source["page"] >= 1, "citations must be 1-based page numbers"
        assert source["excerpt"]

    # The retrieved context and the grounding guardrail both reached the model.
    sent = fake_llm[0]
    assert sent[0]["role"] == "system"
    assert "strictly using the provided context" in sent[0]["content"]
    assert "Context block 1" in sent[-1]["content"]


def test_query_retrieves_at_most_top_k_sources(client, sample_pdf):
    session_id = ingest(client, sample_pdf).json()["session_id"]
    response = client.post(
        "/api/v1/context/query",
        json={"session_id": session_id, "query": "citations"},
    )
    assert len(response.json()["sources"]) <= settings.top_k


def test_follow_up_question_carries_chat_history(client, sample_pdf, fake_llm):
    session_id = ingest(client, sample_pdf).json()["session_id"]
    payload = {"session_id": session_id, "query": "What is Contexto?"}
    client.post("/api/v1/context/query", json=payload)

    client.post(
        "/api/v1/context/query",
        json={"session_id": session_id, "query": "Elaborate on that."},
    )

    second_call = fake_llm[1]
    roles = [message["role"] for message in second_call]
    assert roles == ["system", "user", "assistant", "user"]
    assert second_call[1]["content"] == "What is Contexto?"
    assert second_call[2]["content"] == FAKE_ANSWER


def test_query_rejects_unknown_session(client):
    response = client.post(
        "/api/v1/context/query",
        json={"session_id": "does-not-exist", "query": "hi"},
    )
    assert response.status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "", "query": "hi"},
        {"session_id": "abc", "query": ""},
        {"query": "missing session"},
    ],
)
def test_query_validates_payload(client, payload):
    response = client.post("/api/v1/context/query", json=payload)
    assert response.status_code == 422


def test_provider_failure_surfaces_as_502(client, sample_pdf, monkeypatch):
    session_id = ingest(client, sample_pdf).json()["session_id"]

    def boom(messages):
        raise llm.LLMError("provider is down")

    monkeypatch.setattr(llm, "generate", boom)

    response = client.post(
        "/api/v1/context/query",
        json={"session_id": session_id, "query": "anything"},
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "provider is down"


def test_unexpected_failure_does_not_leak_internals(client, sample_pdf, monkeypatch):
    session_id = ingest(client, sample_pdf).json()["session_id"]

    def boom(messages):
        raise ZeroDivisionError("secret internal detail")

    monkeypatch.setattr(llm, "generate", boom)

    response = client.post(
        "/api/v1/context/query",
        json={"session_id": session_id, "query": "anything"},
    )
    assert response.status_code == 500
    assert "secret internal detail" not in response.text


def test_sessions_are_isolated_from_each_other(client, sample_pdf):
    from tests.conftest import build_pdf

    first = ingest(client, sample_pdf).json()["session_id"]
    other_pdf = build_pdf(["Completely unrelated content about marine biology."])
    second = ingest(client, other_pdf, "other.pdf").json()["session_id"]
    assert first != second

    response = client.post(
        "/api/v1/context/query",
        json={"session_id": second, "query": "What is this about?"},
    )
    excerpts = " ".join(source["excerpt"] for source in response.json()["sources"])
    assert "marine biology" in excerpts
    assert "FastAPI" not in excerpts


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        events.append(
            (lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: ")))
        )
    return events


def test_stream_emits_sources_then_tokens_then_done(client, sample_pdf):
    session_id = ingest(client, sample_pdf).json()["session_id"]

    response = client.post(
        "/api/v1/context/query/stream",
        json={"session_id": session_id, "query": "What is Contexto?"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(response.text)
    names = [name for name, _ in events]

    assert names[0] == "sources", "citations must arrive before generation"
    assert names[-1] == "done"
    assert names.count("token") > 1, "expected an incremental token stream"

    assert events[0][1]["sources"], "sources event should carry citations"
    streamed = "".join(data["t"] for name, data in events if name == "token")
    assert streamed.strip() == FAKE_ANSWER


def test_stream_commits_the_answer_to_chat_memory(client, sample_pdf, fake_llm):
    from app.services import memory

    session_id = ingest(client, sample_pdf).json()["session_id"]
    client.post(
        "/api/v1/context/query/stream",
        json={"session_id": session_id, "query": "What is Contexto?"},
    )

    history = [m.content for m in memory.get_history(session_id).messages]
    assert history == ["What is Contexto?", FAKE_ANSWER]


def test_stream_rejects_unknown_session(client):
    response = client.post(
        "/api/v1/context/query/stream",
        json={"session_id": "nope", "query": "hi"},
    )
    assert response.status_code == 404


def test_stream_reports_provider_failure_as_an_error_event(
    client, sample_pdf, monkeypatch
):
    session_id = ingest(client, sample_pdf).json()["session_id"]

    def boom(messages):
        raise llm.LLMError("provider is down")
        yield  # pragma: no cover - makes this a generator function

    monkeypatch.setattr(llm, "stream", boom)

    response = client.post(
        "/api/v1/context/query/stream",
        json={"session_id": session_id, "query": "anything"},
    )
    # The status line is already sent, so failures ride the stream instead.
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert ("error", {"detail": "provider is down"}) in events


def test_clear_removes_session_and_memory(client, sample_pdf):
    from app.services import memory

    session_id = ingest(client, sample_pdf).json()["session_id"]
    client.post(
        "/api/v1/context/query",
        json={"session_id": session_id, "query": "What is Contexto?"},
    )
    assert session_id in memory._histories

    response = client.request(
        "DELETE", "/api/v1/context/clear", json={"session_id": session_id}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    assert session_id not in memory._histories

    follow_up = client.post(
        "/api/v1/context/query", json={"session_id": session_id, "query": "hi"}
    )
    assert follow_up.status_code == 404


def test_clear_rejects_unknown_session(client):
    response = client.request(
        "DELETE", "/api/v1/context/clear", json={"session_id": "nope"}
    )
    assert response.status_code == 404
