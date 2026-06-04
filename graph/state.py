"""
state.py — Defines the shared state object passed between all LangGraph nodes.

Think of GraphState as a "shared whiteboard" that every agent in the pipeline
can read from and write to. LangGraph passes this dict through the graph,
and each node returns an updated copy.

Each field is set at some point in the pipeline:
  load_memory    → chat_history, memory_context
  route_query    → route
  rewrite_query  → rewritten_query
  retrieve       → child_hits, parent_docs
  grade_documents→ graded_docs
  generate       → generation
  validate_response → validation_result, validation_reason
  save_memory    → (persists turn to Redis, no state change)

TypedDict ensures type checking — Python will warn if you access a
field that doesn't exist in the definition.
"""
from __future__ import annotations

from typing import Dict, List, TypedDict


class GraphState(TypedDict):

    # ── Who is asking ─────────────────────────────────────────────────────
    query:      str   # the user's original question (never modified)
    user_id:    str   # identifies the user for long-term memory
    session_id: str   # unique ID for this conversation session

    # ── Memory (loaded at the start of each graph run) ────────────────────
    chat_history:   List[Dict]   # recent Q&A turns: [{"role": ..., "content": ...}]
    memory_context: str          # compact long-term memory injected into system prompt

    # ── Routing (set by router agent) ─────────────────────────────────────
    route: str   # "rag" → use vector search | "web_search" → use Tavily

    # ── Retrieval pipeline ────────────────────────────────────────────────
    rewritten_query: str         # expanded/clarified version of the query
    child_hits:      List[Dict]  # raw Qdrant search results (child chunks)
    parent_docs:     List[Dict]  # full context fetched from MongoDB
    graded_docs:     List[Dict]  # docs that passed relevance grading

    # ── Generation and validation ─────────────────────────────────────────
    generation:        str   # the LLM's answer
    validation_result: str   # "valid" | "hallucination" | "irrelevant"
    validation_reason: str   # one-sentence explanation from validator
    retry_count:       int   # how many times we've retried generation (max 2)