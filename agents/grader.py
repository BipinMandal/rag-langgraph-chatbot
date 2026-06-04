"""
grader.py — Scores each retrieved document for relevance to the query.

Why grade documents?
Vector search returns the "most similar" chunks, but "similar" doesn't always
mean "useful for answering this specific question". The grader uses an LLM
to make a judgment call.

Example: Query = "what training optimizer was used?"
- A chunk about "Adam optimizer with β1=0.9" → score 1.0 ✓
- A chunk about "the training data was English-German" → score 0.2 ✗ (filtered)

Cross-reference boosting:
If the user asks "explain Figure 1" and a document contains a cross-reference
to Figure 1 (e.g., "as shown in Figure 1, the architecture..."), that document
gets a small score boost — even if the figure description itself scored lower.

FIX APPLIED:
  ✅ _cross_ref_boost now reads from doc["metadata"]["cross_refs"] which is
     correctly populated by chunker.py (was dead code before — always returned 0.0
     because cross_refs were never stored in ParentChunk.metadata).
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import FAST_MODEL, GROQ_API_KEY, RELEVANCE_THRESHOLD
from graph.state import GraphState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a relevance grader for a RAG system about the
"Attention Is All You Need" transformer paper.

Task: Given a user query and a document passage, score how relevant the passage
is to answering the query.

Score from 0.0 to 1.0:
  1.0 → Directly answers the query with specific facts
  0.7 → Closely related, provides important context
  0.5 → Tangentially related, might help partially
  0.2 → Loosely related, unlikely to help
  0.0 → Completely unrelated

Respond with ONLY valid JSON (no markdown, no explanation):
{"score": <float 0.0-1.0>, "reason": "<one sentence>"}
"""


def _cross_ref_boost(doc: dict, query: str) -> float:
    """
    Give a small score boost to documents whose cross-references
    match what the user is asking about.

    Example:
      doc["metadata"]["cross_refs"] = ["see figure 1"]
      query = "explain the transformer architecture in figure 1"
      → boost of +0.05 because "figure 1" appears in both

    FIX: Now correctly reads from doc["metadata"]["cross_refs"]
         which is populated by chunker.py's _emit_parent_and_children().
         Previously, metadata was always {} so this always returned 0.0.
    """
    # FIX: Read from metadata dict (populated in chunker.py)
    cross_refs  = doc.get("metadata", {}).get("cross_refs", [])
    if not cross_refs:
        return 0.0

    query_lower = query.lower()
    for ref in cross_refs:
        # Check if any word from the cross-reference appears in the query
        ref_words = ref.split()
        if any(word in query_lower for word in ref_words if len(word) > 3):
            return 0.05   # small boost — doesn't override the LLM score dramatically

    return 0.0


def _grade_single_document(llm: ChatGroq, query: str, doc: dict) -> float:
    """
    Ask the LLM to score one document's relevance to the query.
    Returns a float between 0.0 and 1.0.
    """
    # Truncate to 1200 chars for the fast model's context window
    content_snippet = doc.get("content", "")[:1200]

    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"User Query: {query}\n\n"
            f"Document:\n{content_snippet}"
        )),
    ])

    # Parse the JSON response
    raw = response.content.strip().strip("`").replace("json", "", 1).strip()
    try:
        result = json.loads(raw)
        score  = float(result.get("score", 0.5))
        # Clamp to valid range
        score  = max(0.0, min(1.0, score))
    except (json.JSONDecodeError, ValueError):
        score = 0.5   # neutral score on parse failure

    # Apply cross-reference boost (capped at 1.0)
    boost = _cross_ref_boost(doc, query)
    score = min(1.0, score + boost)

    return score


def grade_documents(state: GraphState) -> GraphState:
    """
    Grade each retrieved parent document and filter out irrelevant ones.
    Documents below RELEVANCE_THRESHOLD are removed before generation.
    """
    llm   = ChatGroq(model=FAST_MODEL, api_key=GROQ_API_KEY, max_tokens=80, temperature=0)
    query = state.get("rewritten_query") or state["query"]
    docs  = state.get("parent_docs", [])

    graded_docs: List[Dict] = []

    for doc in docs:
        score  = _grade_single_document(llm, query, doc)
        graded = {**doc, "grade": score}

        if score >= RELEVANCE_THRESHOLD:
            graded_docs.append(graded)
            logger.info(f"  ✓ PASS  score={score:.2f}  section='{doc.get('section_header', '')[:40]}'")
        else:
            logger.info(f"  ✗ FAIL  score={score:.2f}  section='{doc.get('section_header', '')[:40]}'")

    logger.info(
        f"Grader: {len(graded_docs)}/{len(docs)} docs passed "
        f"(threshold={RELEVANCE_THRESHOLD})"
    )
    return {**state, "graded_docs": graded_docs}