"""
web_search.py — Performs web search using Tavily API.

Used in two scenarios:
  1. Router chose "web_search" — the query is out-of-domain (not about the paper)
  2. Corrective RAG fallback — the grader found fewer than MIN_GRADED_DOCS
     relevant documents, so we supplement with web results

Why Tavily?
- AI-optimised search (returns cleaner, longer excerpts than Google)
- Free tier: 1000 searches/month
- Simple Python client

FIX APPLIED:
  ✅ Lazy Tavily client initialisation — client created on first use,
     not at module import time. Prevents crash if TAVILY_API_KEY is missing.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from tavily import TavilyClient

from config import TAVILY_API_KEY
from graph.state import GraphState

logger = logging.getLogger(__name__)

# Lazy singleton — created on first use
_tavily_client: Optional[TavilyClient] = None


def _get_tavily() -> TavilyClient:
    """Get (or create) the Tavily client."""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    return _tavily_client


def web_search(state: GraphState) -> GraphState:
    """
    Perform web search and add results to graded_docs.

    When called as corrective RAG, graded_docs already contains
    whatever docs survived grading (possibly 0). We append web results
    so the generator has something to work with.
    """
    query = state.get("rewritten_query") or state["query"]
    logger.info(f"Web search for: '{query}'")

    try:
        results = _get_tavily().search(
            query        = query,
            max_results  = 4,
            search_depth = "advanced",   # "advanced" gives longer excerpts
        )

        web_docs: List[Dict] = [
            {
                "parent_id":      f"web_{i}",          # synthetic ID for web results
                "content":        r.get("content", r.get("snippet", "")),
                "section_header": "Web Search Result",
                "source":         r.get("url", ""),
                "page_number":    0,
                "grade":          0.8,                 # assume web results are relevant
                "metadata":       {},
            }
            for i, r in enumerate(results.get("results", []))
        ]
        logger.info(f"  Got {len(web_docs)} web results.")

    except Exception as error:
        logger.warning(f"  Tavily search failed: {error}")
        web_docs = []

    # Append web results to any existing graded docs (corrective RAG path)
    combined_docs = state.get("graded_docs", []) + web_docs

    return {**state, "graded_docs": combined_docs}