"""
router.py — Classifies the user's query to decide which pipeline to use.

Two routes:
  "rag"        → The question is about the paper → use vector search
  "web_search" → General question → use Tavily web search

Why do we need routing?
If someone asks "what is the capital of France?", we shouldn't waste time
searching the paper's vector store — we know it won't be there.
The router prevents unnecessary computation and gives better answers
for general questions.

FIXES APPLIED:
  ✅ max_tokens increased from 20 to 50 — prevents truncation on "web_search"
  ✅ Markdown code fence stripping — Groq sometimes wraps JSON in ```json ... ```
     even when told not to. We strip this before parsing.
"""
from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import FAST_MODEL, GROQ_API_KEY
from graph.state import GraphState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a query router for a RAG chatbot about the paper
"Attention Is All You Need" (Vaswani et al., 2017) and transformer architectures.

Decide which route to use:
- "rag"        → The question is about: transformers, attention mechanisms,
                  the paper's content, encoder/decoder, positional encoding,
                  training details, BLEU scores, model architecture, or related ML.
- "web_search" → The question is: general knowledge, unrelated to the paper,
                  or cannot be answered from it.

You MUST respond with ONLY valid JSON. No explanation. No markdown. No backticks.
Valid responses:
  {"route": "rag"}
  {"route": "web_search"}
"""


def route_query(state: GraphState) -> GraphState:
    """
    Classify the query and set state["route"].
    Uses the fast 8b model — this is a simple yes/no classification.
    """
    llm = ChatGroq(
        model       = FAST_MODEL,
        api_key     = GROQ_API_KEY,
        max_tokens  = 50,         # FIX: was 20 — too tight for JSON + whitespace
        temperature = 0,          # 0 = deterministic, no creativity needed here
    )

    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=state["query"]),
    ])

    # FIX: Strip markdown code fences before parsing JSON.
    # Groq occasionally wraps output in ```json ... ``` even when told not to.
    raw_content = response.content.strip()
    raw_content = raw_content.strip("`")             # remove backticks
    raw_content = raw_content.replace("json", "", 1).strip()  # remove "json" language tag

    try:
        result = json.loads(raw_content)
        route  = result.get("route", "rag")
    except json.JSONDecodeError:
        # If we still can't parse, default to "rag" (safe fallback)
        logger.warning(f"Router: could not parse JSON response: '{raw_content}'. Defaulting to 'rag'.")
        route = "rag"

    # Validate the value is one we know about
    if route not in ("rag", "web_search"):
        logger.warning(f"Router: unknown route '{route}'. Defaulting to 'rag'.")
        route = "rag"

    logger.info(f"Router → '{route}' for: '{state['query'][:60]}…'")
    return {**state, "route": route}