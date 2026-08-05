"""Live end-to-end smoke test against a real, configured LLM provider.

Usage: ./venv/bin/python scripts/smoke_test.py [--pdf FILE]

Unlike ``pytest`` (which stubs the provider and runs fully offline), this hits
the configured provider for real. Use it to verify credentials, model
availability, and streaming after changing .env or deploying.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from tests.conftest import build_pdf  # noqa: E402

DEMO_PAGES = [
    "Contexto is a retrieval augmented generation service built with FastAPI, "
    "ChromaDB, and locally computed bge-small-en-v1.5 embeddings.",
    "Answers are grounded strictly in the uploaded document and are returned "
    "with page level citations.",
]


def check(label: str, condition: bool) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="PDF to use (default: synthetic demo doc)")
    args = parser.parse_args()

    if args.pdf:
        pdf_bytes = Path(args.pdf).read_bytes()
        name = Path(args.pdf).name
    else:
        pdf_bytes, name = build_pdf(DEMO_PAGES), "demo.pdf"

    print(f"Provider: {settings.llm_provider} -> {settings.resolved_model}")
    print(f"Endpoint: {settings.resolved_base_url}\n")

    with TestClient(app) as client:
        print("1. Ingest")
        start = time.perf_counter()
        response = client.post(
            "/api/v1/context/ingest",
            files={"file": (name, pdf_bytes, "application/pdf")},
        )
        check(f"status 200 (got {response.status_code})", response.status_code == 200)
        body = response.json()
        session_id = body["session_id"]
        print(
            f"        {body['total_chunks']} chunks in "
            f"{time.perf_counter() - start:.2f}s (session {session_id[:8]}…)"
        )

        print("\n2. Grounded query")
        start = time.perf_counter()
        response = client.post(
            "/api/v1/context/query",
            json={"session_id": session_id, "query": "What is Contexto built with?"},
        )
        check(f"status 200 (got {response.status_code})", response.status_code == 200)
        result = response.json()
        check("answer is non-empty", bool(result["answer"].strip()))
        check("citations returned", len(result["sources"]) > 0)
        check(
            "citations use 1-based pages",
            all(source["page"] >= 1 for source in result["sources"]),
        )
        print(f"        answered in {time.perf_counter() - start:.2f}s")
        print(f"        {result['answer'][:200]}")

        print("\n3. Streaming query")
        tokens, events = [], []
        with client.stream(
            "POST",
            "/api/v1/context/query/stream",
            json={"session_id": session_id, "query": "Summarise the document."},
        ) as response:
            check(
                f"status 200 (got {response.status_code})",
                response.status_code == 200,
            )
            event = None
            for line in response.iter_lines():
                if line.startswith("event: "):
                    event = line[len("event: ") :]
                    events.append(event)
                elif line.startswith("data: ") and event == "token":
                    tokens.append(json.loads(line[len("data: ") :])["t"])
                elif line.startswith("data: ") and event == "error":
                    print("        " + json.loads(line[len("data: ") :])["detail"])
        check("no error event", "error" not in events)
        check("sources sent before tokens", events and events[0] == "sources")
        check("tokens streamed", len(tokens) > 1)
        check("stream terminated with done", events[-1] == "done")
        print(f"        {len(tokens)} tokens: {''.join(tokens)[:160]}…")

        print("\n4. Multi-turn memory")
        response = client.post(
            "/api/v1/context/query",
            json={"session_id": session_id, "query": "Elaborate on your last answer."},
        )
        check(f"status 200 (got {response.status_code})", response.status_code == 200)

        print("\n5. Session isolation")
        response = client.post(
            "/api/v1/context/query",
            json={"session_id": "does-not-exist", "query": "hi"},
        )
        check(f"unknown session rejected (got {response.status_code})",
              response.status_code == 404)

        print("\n6. Clear")
        response = client.request(
            "DELETE", "/api/v1/context/clear", json={"session_id": session_id}
        )
        check(f"status 200 (got {response.status_code})", response.status_code == 200)
        response = client.post(
            "/api/v1/context/query", json={"session_id": session_id, "query": "hi"}
        )
        check(f"session gone (got {response.status_code})",
              response.status_code == 404)

    print("\nAll live checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
