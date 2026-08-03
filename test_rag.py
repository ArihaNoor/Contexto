"""End-to-end test of the Contexto API: ingest -> query -> follow-up -> clear.

Run with: ./venv/bin/python test_rag.py
"""

import time

from fastapi.testclient import TestClient

from app.main import app

PDF_PATH = "data/ArihaNoor.pdf"

with TestClient(app) as client:
    # 1. Ingest
    start = time.perf_counter()
    with open(PDF_PATH, "rb") as f:
        response = client.post(
            "/api/v1/context/ingest",
            files={"file": ("ArihaNoor.pdf", f, "application/pdf")},
        )
    assert response.status_code == 200, response.text
    ingest = response.json()
    session_id = ingest["session_id"]
    print(f"[ingest] {ingest['total_chunks']} chunks in "
          f"{time.perf_counter() - start:.2f}s (session {session_id})")

    # 2. Query
    start = time.perf_counter()
    response = client.post(
        "/api/v1/context/query",
        json={"session_id": session_id, "query": "What is the key summary of this PDF?"},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    print(f"[query] answered in {time.perf_counter() - start:.2f}s")
    print("Answer:", result["answer"])
    print("Sources:", [(s["page"], s["excerpt"][:60] + "...") for s in result["sources"]])

    # 3. Multi-turn follow-up (relies on chat memory)
    response = client.post(
        "/api/v1/context/query",
        json={"session_id": session_id, "query": "Can you elaborate on your last answer?"},
    )
    assert response.status_code == 200, response.text
    print("Follow-up answer:", response.json()["answer"][:200], "...")

    # 4. Unknown session is rejected
    response = client.post(
        "/api/v1/context/query",
        json={"session_id": "does-not-exist", "query": "hi"},
    )
    assert response.status_code == 404, response.text
    print("[isolation] unknown session correctly rejected (404)")

    # 5. Clear
    response = client.request(
        "DELETE", "/api/v1/context/clear", json={"session_id": session_id}
    )
    assert response.status_code == 200, response.text
    response = client.post(
        "/api/v1/context/query", json={"session_id": session_id, "query": "hi"}
    )
    assert response.status_code == 404, "session should be gone after clear"
    print("[clear] session flushed successfully")

print("\nAll end-to-end checks passed.")
