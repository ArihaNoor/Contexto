"""Measure ingestion throughput and end-to-end query latency against the live provider.

Usage: ./venv/bin/python scripts/benchmark.py [--pdf path/to/file.pdf] [--runs 5]

Ingestion is measured locally (CPU embeddings, no network). Query latency includes
vector search plus the Hugging Face chat-completion round trip, so results depend
on provider load.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from tests.conftest import build_pdf  # noqa: E402

QUESTIONS = [
    "What is the key summary of this document?",
    "List the main topics covered.",
    "What technologies are mentioned?",
    "Summarise the most important detail on the first page.",
    "What does the document say about results or outcomes?",
]


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(int(round((pct / 100) * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def report(label: str, samples: list[float]) -> None:
    print(
        f"{label:<28} n={len(samples):<3} "
        f"mean={statistics.mean(samples):.2f}s  "
        f"median={statistics.median(samples):.2f}s  "
        f"p95={percentile(samples, 95):.2f}s  "
        f"max={max(samples):.2f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="PDF to benchmark against (default: synthetic)")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    if args.pdf:
        pdf_bytes = Path(args.pdf).read_bytes()
        label = Path(args.pdf).name
    else:
        page = (
            "Contexto indexes documents for retrieval augmented generation. "
            "Chunks are embedded locally and stored in ChromaDB with page metadata. "
        ) * 12
        pdf_bytes = build_pdf([f"Page {i + 1}. {page}" for i in range(20)])
        label = "synthetic-20-page.pdf"

    print(f"Document : {label} ({len(pdf_bytes) / 1024:.0f} KB)")
    print(f"Embedding: {settings.embedding_model} (local CPU)")
    print(f"LLM      : {settings.llm_model}")
    print(f"Retrieval: top_k={settings.top_k}, chunk_size={settings.chunk_size}, "
          f"overlap={settings.chunk_overlap}\n")

    ingest_times: list[float] = []
    query_times: list[float] = []
    total_chunks = 0

    with TestClient(app) as client:
        for run in range(args.runs):
            start = time.perf_counter()
            response = client.post(
                "/api/v1/context/ingest",
                files={"file": (label, pdf_bytes, "application/pdf")},
            )
            response.raise_for_status()
            ingest_times.append(time.perf_counter() - start)

            body = response.json()
            session_id = body["session_id"]
            total_chunks = body["total_chunks"]

            start = time.perf_counter()
            query = client.post(
                "/api/v1/context/query",
                json={"session_id": session_id, "query": QUESTIONS[run % len(QUESTIONS)]},
            )
            query.raise_for_status()
            query_times.append(time.perf_counter() - start)

            client.request(
                "DELETE", "/api/v1/context/clear", json={"session_id": session_id}
            )

    print(f"Chunks indexed per run: {total_chunks}\n")
    report("Ingest + index", ingest_times)
    report("Query (retrieval + LLM)", query_times)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
