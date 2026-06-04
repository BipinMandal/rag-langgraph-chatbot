"""
pipeline.py — Wires all agents into a LangGraph stateful pipeline.

How LangGraph works:
  - NODES  = functions that read state, do work, return updated state
  - EDGES  = connections between nodes (who runs after who)
  - CONDITIONAL EDGES = the next node depends on a value in state

Full flow:
  START
    └─► load_memory
          └─► route_query
                ├─[rag]────► rewrite_query
                │                 └─► retrieve
                │                       └─► grade_documents
                │                             ├─[enough docs]──────────────────┐
                │                             └─[too few]──► web_search ───────┤
                │                                                               │
                └─[web_search]──────────────► web_search ──────────────────────┤
                                                                                │
                                                                          generate
                                                                               │
                                                                    validate_response
                                                                          ├─[valid]──► save_memory ──► END
                                                                          └─[invalid]─► increment_retry
                                                                                              │
                                                                                    (back to generate,
                                                                                     max 2 retries)

FIXES APPLIED:
  ✅ MIN_GRADED_DOCS from config.py (was hardcoded >= 2, now configurable)
  ✅ load_memory resets ALL retrieval fields so stale state never leaks
     between runs (important if graph is ever used in an API/server context)
  ✅ atexit and signal handlers moved to main.py (registered inside chat()
     not at module level)
"""
from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from agents.generator       import generate
from agents.grader          import grade_documents
from agents.query_rewriter  import rewrite_query
from agents.retriever       import retrieve
from agents.router          import route_query
from agents.validator       import validate_response
from agents.web_search      import web_search
from config                 import MAX_VALIDATOR_RETRIES, MIN_GRADED_DOCS
from graph.state            import GraphState
from memory.long_term       import build_memory_prompt
from memory.short_term      import append_turn, load_history

logger = logging.getLogger(__name__)


# ── Node functions ────────────────────────────────────────────────────────────
# Each node is a plain Python function that:
#   1. Receives the current GraphState dict
#   2. Does some work
#   3. Returns an updated GraphState dict (use {**state, "key": new_value})

def load_memory(state: GraphState) -> GraphState:
    """
    First node in the graph — runs before every query.

    Responsibilities:
    1. Load chat history from Redis (short-term memory)
    2. Load the long-term summary for this user
    3. Reset all pipeline fields so stale data from a previous run
       never accidentally leaks into the current run

    FIX: All retrieval/generation fields are now explicitly reset here.
    This makes the graph safe to reuse in a server context where the
    same graph object handles multiple requests.
    """
    history = load_history(state["session_id"])
    mem_ctx = build_memory_prompt(state["user_id"])

    return {
        **state,
        # ── Memory ────────────────────────────────────────
        "chat_history":      history,
        "memory_context":    mem_ctx,
        # ── Reset pipeline state (FIX: was not reset before) ──
        "route":             "",
        "rewritten_query":   "",
        "child_hits":        [],
        "parent_docs":       [],
        "graded_docs":       [],
        "generation":        "",
        "validation_result": "",
        "validation_reason": "",
        "retry_count":       0,
    }


def save_memory(state: GraphState) -> GraphState:
    """
    Last node before END — saves this Q&A turn to Redis.

    Appends:
      {"role": "user",      "content": <original query>}
      {"role": "assistant", "content": <generated answer>}

    The TTL is refreshed on every save so the session stays alive
    as long as the user keeps asking questions.
    """
    append_turn(
        session_id    = state["session_id"],
        user_msg      = state["query"],
        assistant_msg = state["generation"],
    )
    logger.info("Turn saved to short-term memory.")
    return state


def increment_retry(state: GraphState) -> GraphState:
    """
    Bump the retry counter before sending the state back to generate.

    Why a separate node?
    LangGraph needs explicit nodes — we can't just modify a value
    inside a conditional edge function. This node exists purely to
    increment retry_count before the next generate() call.
    """
    new_count = state.get("retry_count", 0) + 1
    logger.info(f"Retry count incremented to {new_count}.")
    return {**state, "retry_count": new_count}


# ── Conditional edge functions ────────────────────────────────────────────────
# These functions look at the current state and return the NAME of the
# next node to visit. LangGraph uses the return value to follow the right edge.

def decide_after_route(
    state: GraphState,
) -> Literal["rewrite_query", "web_search"]:
    """
    After routing: go to RAG pipeline or straight to web search.

    "rag"        → rewrite_query (then retrieve → grade → generate)
    "web_search" → web_search   (then generate directly)
    """
    if state["route"] == "rag":
        return "rewrite_query"
    return "web_search"


def decide_after_grade(
    state: GraphState,
) -> Literal["generate", "web_search"]:
    """
    After grading: do we have enough good documents to generate an answer?

    FIX: Uses MIN_GRADED_DOCS from config.py (was hardcoded as >= 2).
    Now set to 1 — even a single highly-relevant document is enough.

    If fewer than MIN_GRADED_DOCS passed grading:
      → Corrective RAG: fall back to web search to supplement
    Otherwise:
      → Proceed to generate with the graded documents
    """
    num_good_docs = len(state.get("graded_docs", []))

    if num_good_docs >= MIN_GRADED_DOCS:
        logger.info(f"Grading passed: {num_good_docs} docs → proceeding to generate.")
        return "generate"

    logger.info(
        f"Only {num_good_docs} docs passed grading (need {MIN_GRADED_DOCS}). "
        f"Triggering corrective RAG — falling back to web search."
    )
    return "web_search"


def decide_after_validate(
    state: GraphState,
) -> Literal["save_memory", "increment_retry"]:
    """
    After validation: was the answer good enough, or do we retry?

    "valid"   → save to memory and return to user
    anything else + retries remaining → increment_retry → generate again
    anything else + max retries hit   → save anyway (best effort)

    The retry loop works because:
    1. increment_retry bumps retry_count
    2. generate() sees retry_count > 0 and injects the correction hint
    3. validator checks the new answer
    4. This repeats up to MAX_VALIDATOR_RETRIES times
    """
    verdict     = state.get("validation_result", "valid")
    retry_count = state.get("retry_count", 0)

    if verdict == "valid":
        logger.info("Validation passed — saving answer.")
        return "save_memory"

    if retry_count >= MAX_VALIDATOR_RETRIES:
        logger.warning(
            f"Max retries ({MAX_VALIDATOR_RETRIES}) reached. "
            f"Returning best available answer (verdict was '{verdict}')."
        )
        return "save_memory"

    logger.info(
        f"Validation failed ('{verdict}'): {state.get('validation_reason', '')}. "
        f"Retrying ({retry_count + 1}/{MAX_VALIDATOR_RETRIES}) …"
    )
    return "increment_retry"


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph pipeline.

    This is called once at module load time.
    The compiled graph is reused for every user query.
    """
    # Create a new graph with our state type
    graph = StateGraph(GraphState)

    # ── Register all nodes ─────────────────────────────────────────────────
    # Each call to add_node gives a name to a function.
    # The name is what conditional edges return to select the next step.
    graph.add_node("load_memory",        load_memory)
    graph.add_node("route_query",        route_query)
    graph.add_node("rewrite_query",      rewrite_query)
    graph.add_node("retrieve",           retrieve)
    graph.add_node("grade_documents",    grade_documents)
    graph.add_node("web_search",         web_search)
    graph.add_node("generate",           generate)
    graph.add_node("validate_response",  validate_response)
    graph.add_node("increment_retry",    increment_retry)
    graph.add_node("save_memory",        save_memory)

    # ── Wire the edges ─────────────────────────────────────────────────────

    # Entry point
    graph.add_edge(START, "load_memory")

    # Always go from load_memory → route_query
    graph.add_edge("load_memory", "route_query")

    # After routing: branch based on route value
    graph.add_conditional_edges(
        "route_query",          # from this node
        decide_after_route,     # call this function to decide next node
        {
            "rewrite_query": "rewrite_query",   # if returns "rewrite_query"
            "web_search":    "web_search",       # if returns "web_search"
        }
    )

    # RAG path: rewrite → retrieve → grade
    graph.add_edge("rewrite_query",   "retrieve")
    graph.add_edge("retrieve",        "grade_documents")

    # After grading: enough docs? → generate; too few? → web_search
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grade,
        {
            "generate":   "generate",
            "web_search": "web_search",
        }
    )

    # Both web_search paths (direct + corrective RAG) lead to generate
    graph.add_edge("web_search", "generate")

    # After generating, always validate
    graph.add_edge("generate", "validate_response")

    # After validation: save if valid/max-retries, else retry
    graph.add_conditional_edges(
        "validate_response",
        decide_after_validate,
        {
            "save_memory":     "save_memory",
            "increment_retry": "increment_retry",
        }
    )

    # Retry loop: increment counter → back to generate
    graph.add_edge("increment_retry", "generate")

    # Final node → END
    graph.add_edge("save_memory", END)

    # Compile the graph (validates structure, checks for unreachable nodes, etc.)
    return graph.compile()


# ── Module-level compiled graph ───────────────────────────────────────────────
# Build once when this module is imported.
# Import this in main.py:  from graph.pipeline import rag_graph
rag_graph = build_graph()
logger.info("LangGraph RAG pipeline compiled and ready.")