import json
import logging
import time
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.config import settings
from app.schemas import (
    ClearRequest,
    ClearResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.services import memory
from app.services.ingestion import ingest_pdf
from app.services.llm import LLMError
from app.services.rag import answer_query, stream_answer
from app.services.vectorstore import delete_session, session_exists

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/context", tags=["context"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile) -> IngestResponse:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    file_bytes = await file.read()
    if len(file_bytes) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_file_size_mb}MB size limit.",
        )
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    start = time.perf_counter()
    try:
        session_id, total_chunks = await run_in_threadpool(
            ingest_pdf, file_bytes, file.filename
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.info(
        "ingested %s (%d chunks) into session %s in %.2fs",
        file.filename,
        total_chunks,
        session_id,
        time.perf_counter() - start,
    )
    return IngestResponse(session_id=session_id, total_chunks=total_chunks)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    if not session_exists(request.session_id):
        raise HTTPException(status_code=404, detail="Unknown session_id.")

    start = time.perf_counter()
    try:
        answer, sources = await run_in_threadpool(
            answer_query, request.session_id, request.query
        )
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Query failed for session %s", request.session_id)
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while answering. Please try again.",
        ) from exc

    logger.info(
        "answered session %s in %.2fs (%d sources)",
        request.session_id,
        time.perf_counter() - start,
        len(sources),
    )
    return QueryResponse(answer=answer, sources=sources)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/query/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    """Same RAG pipeline as ``/query``, delivered as server-sent events.

    Citations arrive in a single ``sources`` event before generation starts, so
    the client can render them while tokens are still streaming in. Errors after
    the response has begun are delivered as an ``error`` event rather than an
    HTTP status, since the status line is already on the wire.
    """
    if not session_exists(request.session_id):
        raise HTTPException(status_code=404, detail="Unknown session_id.")

    start = time.perf_counter()

    def events() -> Iterator[str]:
        try:
            sources, tokens = stream_answer(request.session_id, request.query)
            yield _sse("sources", {"sources": [s.model_dump() for s in sources]})

            first_token_at = None
            for token in tokens:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                yield _sse("token", {"t": token})

            logger.info(
                "streamed session %s — first token in %.2fs, complete in %.2fs",
                request.session_id,
                (first_token_at or time.perf_counter()) - start,
                time.perf_counter() - start,
            )
            yield _sse("done", {})
        except LLMError as exc:
            yield _sse("error", {"detail": str(exc)})
        except Exception:
            logger.exception("Streaming query failed for %s", request.session_id)
            yield _sse(
                "error",
                {"detail": "Something went wrong while answering. Please try again."},
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/clear", response_model=ClearResponse)
async def clear(request: ClearRequest) -> ClearResponse:
    if not session_exists(request.session_id):
        raise HTTPException(status_code=404, detail="Unknown session_id.")

    delete_session(request.session_id)
    memory.clear_history(request.session_id)
    return ClearResponse()
