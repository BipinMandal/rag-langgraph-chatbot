"""
validator.py — Checks the generated answer for quality issues.

Three possible verdicts:
  "valid"         → Answer is grounded in context AND addresses the query ✓
  "hallucination" → Answer contains facts NOT found in the context
  "irrelevant"    → Answer doesn't address what was asked

If the verdict is not "valid", the pipeline retries generation (up to
MAX_VALIDATOR_RETRIES times) with the reason injected into the prompt.

The validation_reason field (FIX applied previously) is critical for
the retry loop — without it, the generator would produce identical answers.
"""
from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import FAST_MODEL, GROQ_API_KEY
from graph.state import GraphState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a strict quality checker for a RAG chatbot answer.

You will be given:
1. The user's question
2. The context documents used to generate the answer
3. The generated answer

Evaluate the answer and respond with ONLY valid JSON (no markdown):
{
  "verdict": "valid" | "hallucination" | "irrelevant",
  "reason":  "<one sentence explaining your verdict>"
}

Definitions:
  "valid"         → The answer is supported by the context AND answers the question.
  "hallucination" → The answer states facts not found in the provided context.
  "irrelevant"    → The answer does not address the user's question at all.

Be strict about hallucinations. If the answer claims a specific number, equation,
or citation that is not in the context, call it a hallucination.
"""


def validate_response(state: GraphState) -> GraphState:
    """
    Validate the generated answer and set validation_result + validation_reason.
    These are used by the pipeline's retry logic.
    """
    llm = ChatGroq(
        model       = FAST_MODEL,
        api_key     = GROQ_API_KEY,
        max_tokens  = 150,
        temperature = 0,
    )

    # Summarise context (first 400 chars per doc, max 2000 total)
    # We truncate to stay within the fast model's limits
    context_summary = "\n---\n".join(
        doc.get("content", "")[:400]
        for doc in state.get("graded_docs", [])
    )[:2000]

    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Question: {state['query']}\n\n"
            f"Context (truncated):\n{context_summary}\n\n"
            f"Generated Answer:\n{state['generation']}"
        )),
    ])

    # Parse the JSON response (strip code fences just in case)
    raw = response.content.strip().strip("`").replace("json", "", 1).strip()
    try:
        result  = json.loads(raw)
        verdict = result.get("verdict", "valid")
        reason  = result.get("reason",  "no reason given")
    except (json.JSONDecodeError, AttributeError):
        verdict = "valid"   # safe default — better to return a potentially imperfect
        reason  = "JSON parse error in validator — defaulting to valid"

    # Validate verdict is one of the three known values
    if verdict not in ("valid", "hallucination", "irrelevant"):
        verdict = "valid"

    logger.info(f"Validator → {verdict}: {reason}")
    return {**state, "validation_result": verdict, "validation_reason": reason}