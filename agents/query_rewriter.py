"""
query_rewriter.py — Improves the user's query before vector search.
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import FAST_MODEL, GROQ_API_KEY
from graph.state import GraphState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a search query optimizer for a RAG system over the
"Attention Is All You Need" paper (Vaswani et al., 2017).

Your task: rewrite the user's query to improve vector search retrieval.

Rules:
1. Expand abbreviations:
   - "MHA" → "multi-head attention"
   - "FFN" → "feed-forward network"
   - "PE"  → "positional encoding"
2. Add technical terms the paper uses (e.g., "query", "key", "value", "head")
3. If the query is already specific and technical, return it unchanged
4. Output ONLY the rewritten query — no explanation, no quotes, no prefix
"""


def rewrite_query(state: GraphState) -> GraphState:
    """
    Rewrite the user's query for better retrieval.
    Sets state["rewritten_query"].
    """
    llm = ChatGroq(
        model       = FAST_MODEL,
        api_key     = GROQ_API_KEY,
        max_tokens  = 150,
        temperature = 0.2,
    )

    response  = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=state["query"]),
    ])

    rewritten = response.content.strip().strip('"').strip("'")
    if not rewritten:
        rewritten = state["query"]

    logger.info(f"Query rewritten: '{state['query']}' → '{rewritten}'")
    return {**state, "rewritten_query": rewritten}