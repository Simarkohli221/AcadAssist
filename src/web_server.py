"""
web_server.py
-------------
FastAPI server that exposes the RAKA chat endpoint and serves the web UI.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from retriever import Retriever
from generator import Generator
from query_normalizer import normalize_query
from reranker import get_reranker

logger = logging.getLogger("raka.web")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

APP_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = APP_ROOT / "web"
DEFAULT_INDEX_DIR = APP_ROOT / "processed"
INDEX_DIR = Path(os.getenv("RAKA_INDEX_DIR", DEFAULT_INDEX_DIR))

app = FastAPI(title="RAKA Web", version="1.0.0")

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    model: str
    backend: str
    response_time_ms: int
    mock: bool


def _load_components() -> None:
    if not INDEX_DIR.exists():
        logger.warning("Index directory not found: %s", INDEX_DIR)
        app.state.retriever = None
        app.state.reranker = None
        app.state.generator = None
        return

    try:
        retriever = Retriever.load(INDEX_DIR)
        reranker = get_reranker(retriever._embedder)
        generator = Generator()
    except Exception as exc:
        logger.exception("Failed to load RAKA components: %s", exc)
        app.state.retriever = None
        app.state.reranker = None
        app.state.generator = None
        return

    app.state.retriever = retriever
    app.state.reranker = reranker
    app.state.generator = generator
    logger.info("RAKA components loaded from %s", INDEX_DIR)


@app.on_event("startup")
def startup_event() -> None:
    _load_components()


@app.get("/")
def index() -> FileResponse:
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index_file)


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    retriever = getattr(app.state, "retriever", None)
    reranker = getattr(app.state, "reranker", None)
    generator = getattr(app.state, "generator", None)

    if retriever is None or reranker is None or generator is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Index not loaded. Build the index first and set RAKA_INDEX_DIR "
                "to the processed folder containing faiss.index/metadata." 
            ),
        )

    normalized_query = normalize_query(question)

    results = retriever.retrieve(normalized_query, k=7)
    results = reranker.rerank(normalized_query, results, top_k=7)

    response = generator.generate(normalized_query, results)

    return ChatResponse(
        answer=response.answer,
        sources=response.sources,
        model=response.model,
        backend=response.backend,
        response_time_ms=response.response_time_ms,
        mock=response.mock,
    )
