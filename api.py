"""
api.py — FastAPI backend that exposes the LangGraph pipeline over HTTP.

The Streamlit UI talks to this API — they run as two separate processes.
This separation means:
  - The heavy graph/model loading happens once in the API process
  - The UI stays lightweight and fast
  - You could swap the UI for a mobile app, Discord bot, etc.

Endpoints:
  POST /chat      — send a query, get back answer + citations
  POST /end       — end session (save long-term memory)
  GET  /health    — check the API is running

Run with:
  uv run uvicorn api:app --reload --port 8000
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the compiled graph — this is the expensive operation,
# done once when the API server starts
from graph.pipeline   import rag_graph
from memory.long_term import summarize_and_save

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title       = "RAG Chatbot API",
    description = "LangGraph RAG pipeline over the Attention Is All You Need paper",
    version     = "1.0.0",
)

# Allow the Streamlit UI (running on port 8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query:      str
    user_id:    str
    session_id: str


class CitedSource(BaseModel):
    doc_number:     int
    section_header: str
    page_number:    int
    element_type:   str          # "text" | "image" | "table" | "equation"
    content:        str
    caption:        str
    source_url:     str
    image_path:     str
    image_base64:   str          # base64-encoded image bytes (empty if not an image)
    table_markdown: str          # markdown table string (empty if not a table)


class ChatResponse(BaseModel):
    answer:            str
    cited_sources:     List[CitedSource]
    route:             str        # "rag" or "web_search"
    rewritten_query:   str
    validation_result: str
    retry_count:       int
    session_id:        str


class EndSessionRequest(BaseModel):
    user_id:    str
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Simple health check — confirms the API is running."""
    return {"status": "ok", "message": "RAG Chatbot API is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Process one user query through the full LangGraph pipeline.

    The pipeline runs synchronously — the response is returned when
    the full answer (including validation) is ready.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    logger.info(f"[{request.session_id[:8]}] Query: {request.query[:80]}")

    # Build the initial state — all fields must be present
    initial_state = {
        "query":             request.query,
        "user_id":           request.user_id,
        "session_id":        request.session_id,
        "chat_history":      [],
        "memory_context":    "",
        "route":             "",
        "rewritten_query":   "",
        "child_hits":        [],
        "parent_docs":       [],
        "graded_docs":       [],
        "generation":        "",
        "validation_result": "",
        "validation_reason": "",
        "retry_count":       0,
        "cited_sources":     [],
    }

    try:
        final_state = rag_graph.invoke(initial_state)
    except Exception as error:
        logger.error(f"Pipeline error: {error}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(error)}")

    # Convert cited_sources dicts to CitedSource Pydantic models
    cited = [
        CitedSource(
            doc_number     = s.get("doc_number", 0),
            section_header = s.get("section_header", ""),
            page_number    = s.get("page_number", 0),
            element_type   = s.get("element_type", "text"),
            content        = s.get("content", ""),
            caption        = s.get("caption", ""),
            source_url     = s.get("source_url", ""),
            image_path     = s.get("image_path", ""),
            image_base64   = s.get("image_base64", ""),
            table_markdown = s.get("table_markdown", ""),
        )
        for s in final_state.get("cited_sources", [])
    ]

    return ChatResponse(
        answer            = final_state.get("generation", "No answer generated."),
        cited_sources     = cited,
        route             = final_state.get("route", ""),
        rewritten_query   = final_state.get("rewritten_query", ""),
        validation_result = final_state.get("validation_result", ""),
        retry_count       = final_state.get("retry_count", 0),
        session_id        = request.session_id,
    )


@app.post("/end-session")
def end_session(request: EndSessionRequest):
    """
    End a session and save long-term memory.
    Call this when the user logs out or closes the chat.
    """
    try:
        summarize_and_save(
            user_id    = request.user_id,
            session_id = request.session_id,
        )
        return {"status": "ok", "message": "Session memory saved."}
    except Exception as error:
        logger.warning(f"Memory save failed: {error}")
        return {"status": "warning", "message": str(error)}