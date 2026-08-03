import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import settings
from app.services.vectorstore import get_embeddings

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )
    # Warm the embedding model at startup so the first ingest/query is fast
    get_embeddings()
    logger.info("Contexto ready — embedding model '%s' warm", settings.embedding_model)
    yield


app = FastAPI(
    title="Contexto",
    description="Deep document understanding, simplified. "
    "Upload a PDF and ask context-grounded questions with page-level citations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["health"])
async def health() -> dict:
    return {"app": "Contexto", "status": "ok"}
