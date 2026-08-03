"""Shared pytest fixtures.

The suite runs fully offline: the Hugging Face text-generation call is the only
network dependency and it is monkeypatched out. Embeddings run locally through
fastembed's ONNX runtime, and ChromaDB is redirected to a per-test temp dir so
the developer's real ``chroma_db/`` is never touched.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.services import llm, memory, vectorstore

FAKE_ANSWER = "The document describes a RAG pipeline built on FastAPI."


def build_pdf(pages: list[str]) -> bytes:
    """Build a minimal, valid multi-page PDF with the given text per page.

    Hand-rolled so the test suite needs no PDF-authoring dependency.
    """
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)  # 1-based object number

    catalog_num = 1
    pages_num = 2
    objects.append(b"")  # placeholder for catalog
    objects.append(b"")  # placeholder for pages tree
    font_num = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    kids = []
    for text in pages:
        escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
        content_num = add(
            b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)
        )
        page_num = add(
            b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
            b"/Contents %d 0 R /Resources << /Font << /F1 %d 0 R >> >> >>"
            % (pages_num, content_num, font_num)
        )
        kids.append(page_num)

    objects[catalog_num - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % pages_num
    objects[pages_num - 1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % k for k in kids),
        len(kids),
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)

    xref_offset = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        catalog_num,
        xref_offset,
    )
    return bytes(out)


@pytest.fixture()
def sample_pdf() -> bytes:
    return build_pdf(
        [
            "Contexto is a retrieval augmented generation service built with FastAPI.",
            "Embeddings are produced locally by the bge-small-en-v1.5 model.",
            "Every answer is returned with page level source citations.",
        ]
    )


@pytest.fixture()
def blank_pdf() -> bytes:
    """A structurally valid PDF that contains no extractable text."""
    return build_pdf([" "])


@pytest.fixture(autouse=True)
def isolated_chroma(tmp_path, monkeypatch):
    """Point ChromaDB at a throwaway directory for every test."""
    monkeypatch.setattr(settings, "chroma_dir", str(tmp_path / "chroma"))
    vectorstore.get_client.cache_clear()
    yield
    vectorstore.get_client.cache_clear()


@pytest.fixture(autouse=True)
def clean_memory():
    memory._histories.clear()
    yield
    memory._histories.clear()


@pytest.fixture()
def fake_llm(monkeypatch):
    """Replace the provider call and record the messages it receives."""
    calls: list[list[dict]] = []

    def _generate(messages: list[dict]) -> str:
        calls.append(messages)
        return FAKE_ANSWER

    monkeypatch.setattr(llm, "generate", _generate)
    return calls


@pytest.fixture()
def client(fake_llm):
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
