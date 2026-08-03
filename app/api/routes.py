import logging
import time

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

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
from app.services.rag import answer_query
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


@router.delete("/clear", response_model=ClearResponse)
async def clear(request: ClearRequest) -> ClearResponse:
    if not session_exists(request.session_id):
        raise HTTPException(status_code=404, detail="Unknown session_id.")

    delete_session(request.session_id)
    memory.clear_history(request.session_id)
    return ClearResponse()
