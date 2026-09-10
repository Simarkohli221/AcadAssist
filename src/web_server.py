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
from faithfulness import FaithfulnessVerifier
from reranker import get_reranker
from pipeline import run_pipeline

logger = logging.getLogger("raka.web")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

APP_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = APP_ROOT / "web"
DEFAULT_INDEX_DIR = APP_ROOT / "data" / "processed"
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
    faithfulness_status: str
    faithfulness_score: float
    faithfulness_features: dict[str, float]
    warning: str | None = None


def _load_components() -> None:
    if not INDEX_DIR.exists():
        logger.warning("Index directory not found: %s", INDEX_DIR)
        app.state.retriever = None
        app.state.reranker = None
        app.state.generator = None
        app.state.verifier = None
        return

    try:
        retriever = Retriever.load(INDEX_DIR)
        reranker = get_reranker(retriever._embedder)
        generator = Generator()
        # Reuse retriever._embedder so the faithfulness verifier never loads
        # a second SBERT instance, and so it embeds with the exact same
        # backend used for retrieval (same semantics as chat.py).
        verifier = FaithfulnessVerifier(embedder=retriever._embedder, index_dir=INDEX_DIR)
    except Exception as exc:
        logger.exception("Failed to load RAKA components: %s", exc)
        app.state.retriever = None
        app.state.reranker = None
        app.state.generator = None
        app.state.verifier = None
        return

    app.state.retriever = retriever
    app.state.reranker = reranker
    app.state.generator = generator
    app.state.verifier = verifier
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
    verifier = getattr(app.state, "verifier", None)

    if retriever is None or reranker is None or generator is None or verifier is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Index or faithfulness model not loaded. Build the index "
                "(main.py build) and train the faithfulness model "
                "(create_faithfulness_dataset.py && train_faithfulness.py) "
                "first, and set RAKA_INDEX_DIR to the processed folder "
                "containing faiss.index/metadata."
            ),
        )

    try:
        result = run_pipeline(question, retriever, reranker, generator, verifier)
    except Exception as exc:
        logger.exception("Pipeline failed for question=%r", question)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer. Please try again.",
        ) from exc

    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        model=result.model,
        backend=result.backend,
        response_time_ms=result.response_time_ms,
        mock=result.mock,
        faithfulness_status=result.faithfulness_status,
        faithfulness_score=result.faithfulness_score,
        faithfulness_features=result.faithfulness_features,
        warning=result.warning,
    )
