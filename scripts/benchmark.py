"""Benchmark ingestion throughput and query latency against a live server.

Usage: ./venv/bin/python scripts/benchmark.py [--pdf FILE] [--runs 5] [--port 8099]

A real uvicorn process is started rather than using FastAPI's TestClient,
because TestClient buffers ``text/event-stream`` responses and would report a
meaningless time-to-first-token.

Ingestion is CPU-local (embedding + indexing, no network). Query latency
includes the provider round trip, so numbers move with provider and hardware.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
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
    if not samples:
        print(f"{label:<32} (no samples)")
        return
    print(
        f"{label:<32} n={len(samples):<3} "
        f"mean={statistics.mean(samples):.2f}s  "
        f"median={statistics.median(samples):.2f}s  "
        f"p95={percentile(samples, 95):.2f}s  "
        f"max={max(samples):.2f}s"
    )


def wait_for_server(base_url: str, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{base_url}/health", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError("server did not become healthy in time")


def stream_query(client: httpx.Client, session_id: str, question: str):
    """Return (time_to_first_token, total_time, token_count)."""
    start = time.perf_counter()
    first: float | None = None
    tokens = 0
    event = None

    with client.stream(
        "POST",
        "/api/v1/context/query/stream",
        json={"session_id": session_id, "query": question},
        timeout=300.0,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: ") and event == "token":
                if first is None:
                    first = time.perf_counter() - start
                tokens += 1
            elif line.startswith("data: ") and event == "error":
                raise RuntimeError(json.loads(line[len("data: ") :])["detail"])

    return first, time.perf_counter() - start, tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="PDF to benchmark against (default: synthetic)")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--port", type=int, default=8099)
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

    base_url = f"http://127.0.0.1:{args.port}"
    print(f"Document : {label} ({len(pdf_bytes) / 1024:.0f} KB)")
    print(f"Embedding: {settings.embedding_model} (local CPU)")
    print(f"Provider : {settings.llm_provider} -> {settings.resolved_model}")
    print(
        f"Retrieval: top_k={settings.top_k}, chunk_size={settings.chunk_size}, "
        f"overlap={settings.chunk_overlap}\n"
    )

    server = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "app.main:app",
            "--port", str(args.port), "--log-level", "warning",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )

    ingest_times: list[float] = []
    ttft_times: list[float] = []
    total_times: list[float] = []
    blocking_times: list[float] = []
    total_chunks = 0

    try:
        wait_for_server(base_url)
        with httpx.Client(base_url=base_url, timeout=300.0) as client:
            # One warm-up round so model load time is not charged to run 1.
            warm = client.post(
                "/api/v1/context/ingest",
                files={"file": (label, pdf_bytes, "application/pdf")},
            ).json()["session_id"]
            stream_query(client, warm, "warm up")
            client.request(
                "DELETE", "/api/v1/context/clear", json={"session_id": warm}
            )

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
                question = QUESTIONS[run % len(QUESTIONS)]

                first, total, _ = stream_query(client, session_id, question)
                if first is not None:
                    ttft_times.append(first)
                total_times.append(total)

                start = time.perf_counter()
                client.post(
                    "/api/v1/context/query",
                    json={"session_id": session_id, "query": question},
                ).raise_for_status()
                blocking_times.append(time.perf_counter() - start)

                client.request(
                    "DELETE", "/api/v1/context/clear", json={"session_id": session_id}
                )
    finally:
        server.terminate()
        server.wait(timeout=30)

    print(f"Chunks indexed per run: {total_chunks}\n")
    report("Ingest + index", ingest_times)
    report("Stream: time to first token", ttft_times)
    report("Stream: full answer", total_times)
    report("Blocking /query: full answer", blocking_times)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
