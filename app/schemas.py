from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    session_id: str
    total_chunks: int


class QueryRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)


class Source(BaseModel):
    page: int
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class ClearRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


class ClearResponse(BaseModel):
    status: str = "success"
