"""Unit tests for the ingestion, memory, and provider layers."""

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


def test_missing_token_raises_actionable_llm_error(monkeypatch):
    monkeypatch.setattr(settings, "huggingfacehub_api_token", "")
    llm.get_llm_client.cache_clear()

    with pytest.raises(llm.LLMError, match="HUGGINGFACEHUB_API_TOKEN"):
        llm.generate([{"role": "user", "content": "hi"}])

    llm.get_llm_client.cache_clear()


def test_unsupported_model_error_names_the_override(monkeypatch):
    class FailingClient:
        def chat_completion(self, **kwargs):
            raise RuntimeError("{'code': 'model_not_supported'}")

    monkeypatch.setattr(llm, "get_llm_client", lambda: FailingClient())

    with pytest.raises(llm.LLMError, match="LLM_MODEL"):
        llm.generate([{"role": "user", "content": "hi"}])


def test_empty_completion_is_rejected(monkeypatch):
    class EmptyClient:
        def chat_completion(self, **kwargs):
            class Message:
                content = ""

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    monkeypatch.setattr(llm, "get_llm_client", lambda: EmptyClient())

    with pytest.raises(llm.LLMError, match="empty response"):
        llm.generate([{"role": "user", "content": "hi"}])
