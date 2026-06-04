"""
retriever.py — Retrieves relevant content using hybrid search.

Two-stage retrieval:
  Stage 1: Search Qdrant (child chunks) using hybrid vector search
           → returns up to TOP_K_CHILD small precise chunks
  Stage 2: Deduplicate by parent_id, fetch TOP_K_FINAL parent docs from MongoDB
           → returns full context passages for the LLM

Why deduplicate?
Multiple child chunks might point to the same parent.
We want unique parents for the LLM — no repeated context.
We keep the parent with the highest child retrieval score.

FIX APPLIED:
  ✅ section_filter is now wired up — if the rewritten query mentions a
     specific section (e.g., "Section 3.2"), that section name is passed
     to Qdrant to restrict the search to that section only.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from config import TOP_K_CHILD, TOP_K_FINAL
from document_store.mongodb_store import MongoParentStore
from graph.state import GraphState
from vectorstore.qdrant_store import QdrantHybridStore

logger = logging.getLogger(__name__)

# Module-level singletons — created once, reused on every query
# (Creating Qdrant/MongoDB clients on every call would be slow)
_qdrant_store: Optional[QdrantHybridStore] = None
_mongo_store:  Optional[MongoParentStore]  = None


def _get_stores() -> tuple[QdrantHybridStore, MongoParentStore]:
    """Lazily initialise store connections."""
    global _qdrant_store, _mongo_store
    if _qdrant_store is None:
        _qdrant_store = QdrantHybridStore()
    if _mongo_store is None:
        _mongo_store = MongoParentStore()
    return _qdrant_store, _mongo_store


# FIX: Pattern to detect section references in the query
# e.g., "Section 3.2", "section 4", "in 3.1"
_SECTION_PATTERN = re.compile(
    r"\b(section|in)\s+(\d+(?:\.\d+)*)\b",
    re.IGNORECASE,
)


def _extract_section_hint(query: str) -> Optional[str]:
    """
    If the query explicitly references a section (e.g., "explain Section 3.2"),
    return that section string to use as a Qdrant filter.
    Returns None if no section reference is found.
    """
    match = _SECTION_PATTERN.search(query)
    if match:
        return match.group(2)   # e.g., "3.2"
    return None


def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve the most relevant parent documents for the user's query.
    """
    qdrant, mongo = _get_stores()

    # Use the rewritten query if available (better than the raw query)
    query = state.get("rewritten_query") or state["query"]

    # FIX: Check if the query mentions a specific section → pass as filter
    section_hint = _extract_section_hint(query)
    if section_hint:
        logger.info(f"  Section filter detected: '{section_hint}'")

    # Stage 1: Hybrid search in Qdrant → get child chunks
    child_hits: List[Dict] = qdrant.hybrid_search(
        query          = query,
        top_k          = TOP_K_CHILD,
        section_filter = section_hint,   # FIX: was always None before
    )
    logger.info(f"Qdrant returned {len(child_hits)} child hits.")

    # Stage 2: Deduplicate by parent_id — keep highest-scoring child per parent
    # Example: if 3 children all point to parent_id "abc", keep only "abc" with max score
    best_score_per_parent: Dict[str, float] = {}
    for hit in child_hits:
        pid   = hit["parent_id"]
        score = hit.get("score", 0.0)
        if pid not in best_score_per_parent or score > best_score_per_parent[pid]:
            best_score_per_parent[pid] = score

    # Rank parents by their best child score, take top N
    top_parent_ids = sorted(
        best_score_per_parent.keys(),
        key   = lambda pid: best_score_per_parent[pid],
        reverse = True,
    )[:TOP_K_FINAL]

    # Stage 3: Fetch full parent documents from MongoDB
    parent_docs = mongo.get_parents_by_ids(top_parent_ids)

    # Attach retrieval score to each parent doc (used by grader for context)
    for doc in parent_docs:
        doc["retrieval_score"] = best_score_per_parent.get(doc["parent_id"], 0.0)

    logger.info(f"Retrieved {len(parent_docs)} parent docs from MongoDB.")
    return {**state, "child_hits": child_hits, "parent_docs": parent_docs}