"""PDF ingestion: load, split, embed, and index into a per-session collection."""

import tempfile
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.services.vectorstore import get_vectorstore


def ingest_pdf(file_bytes: bytes, filename: str) -> tuple[str, int]:
    """Index a PDF into a new isolated session collection.

    Returns (session_id, total_chunks).
    """
    session_id = uuid.uuid4().hex

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            raise ValueError("No extractable text found in the PDF.")

        for chunk in chunks:
            chunk.metadata["source"] = filename
            # pypdf pages are 0-indexed; store 1-based for human-readable citations
            chunk.metadata["page"] = int(chunk.metadata.get("page", 0)) + 1

        get_vectorstore(session_id).add_documents(chunks)
        return session_id, len(chunks)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
