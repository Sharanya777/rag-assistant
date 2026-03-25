"""
FastAPI Backend — AI Knowledge Assistant
Exposes REST endpoints for document ingestion and semantic Q&A
"""

import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_engine import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL       = os.getenv("LLM_MODEL", "mock")
VECTOR_STORE    = os.getenv("VECTOR_STORE_PATH", "./vector_store")
TOP_K           = int(os.getenv("TOP_K", "5"))
UPLOAD_DIR      = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── App lifecycle ─────────────────────────────────────────────────────────────
rag: Optional[RAGPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    logger.info("Initialising RAG pipeline...")
    rag = RAGPipeline(
        embedding_model=EMBEDDING_MODEL,
        llm_model=LLM_MODEL,
        vector_store_path=VECTOR_STORE,
        top_k=TOP_K,
    )
    logger.info("RAG pipeline ready ✓")
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="AI Knowledge Assistant API",
    description="Production-grade RAG system: upload documents, ask questions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, example="What is machine learning?")
    top_k: Optional[int] = Field(None, ge=1, le=20)

class SourceChunk(BaseModel):
    content: str
    source: str
    page: int
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    query: str
    model_used: str
    chunks_retrieved: int

class IngestResponse(BaseModel):
    status: str
    documents_ingested: int
    chunks_created: int
    total_vectors: int
    filenames: List[str]

class StatsResponse(BaseModel):
    total_vectors: int
    embedding_model: str
    embedding_dim: int
    llm_model: str
    top_k: int
    score_threshold: float
    status: str

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {"message": "AI Knowledge Assistant API", "status": "running", "version": "1.0.0"}

@app.get("/health", tags=["Health"])
async def health():
    stats = rag.get_stats()
    return {"status": "healthy", "vectors_indexed": stats["total_vectors"]}

@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    s = rag.get_stats()
    return StatsResponse(**s, status="operational")

@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_documents(files: List[UploadFile] = File(...)):
    """
    Upload one or more documents (PDF, TXT, MD).
    They are chunked, embedded, and stored in the FAISS index.
    """
    allowed_types = {".pdf", ".txt", ".md"}
    saved_paths = []
    filenames = []

    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix}'. Allowed: {allowed_types}"
            )
        dest = UPLOAD_DIR / upload.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved_paths.append(str(dest))
        filenames.append(upload.filename)
        logger.info(f"Saved upload: {dest}")

    result = rag.ingest(saved_paths)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    return IngestResponse(
        status=result["status"],
        documents_ingested=result["documents_ingested"],
        chunks_created=result["chunks_created"],
        total_vectors=result["total_vectors"],
        filenames=filenames,
    )

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(req: QueryRequest):
    """
    Ask a natural-language question.
    Returns an LLM-generated answer grounded in your documents.
    """
    if req.top_k:
        rag.top_k = req.top_k

    response = rag.query(req.question)

    return QueryResponse(
        answer=response.answer,
        sources=[
            SourceChunk(
                content=s.content[:500],  # truncate for API response
                source=s.source,
                page=s.page,
                score=round(s.score, 4),
            )
            for s in response.sources
        ],
        query=response.query,
        model_used=response.model_used,
        chunks_retrieved=len(response.sources),
    )

@app.delete("/index", tags=["Documents"])
async def clear_index():
    """Wipe the vector store and start fresh."""
    global rag
    import faiss as _faiss
    rag.vector_store.index = _faiss.IndexFlatIP(rag.vector_store.embedding_dim)
    rag.vector_store.documents = []
    rag.vector_store.is_built = False
    # Remove persisted store
    vs_path = Path(VECTOR_STORE)
    for f in vs_path.glob("*"):
        f.unlink()
    return {"status": "cleared", "message": "Vector index has been reset."}

@app.get("/documents", tags=["Documents"])
async def list_documents():
    """List all ingested source documents."""
    if not rag.vector_store.is_built:
        return {"documents": [], "total_chunks": 0}
    sources = {}
    for doc in rag.vector_store.documents:
        name = doc.metadata.get("source_file", doc.metadata.get("source", "unknown"))
        sources[name] = sources.get(name, 0) + 1
    return {
        "documents": [{"filename": k, "chunks": v} for k, v in sources.items()],
        "total_chunks": sum(sources.values()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
