"""
long_term.py — Summarised cross-session memory stored in Upstash Redis.

What is long-term memory?
When a user starts a new session (e.g., next day), they don't want to
repeat context they've already established. Long-term memory solves this
by storing a compact summary of key insights from previous sessions.

Why NOT store the full chat history?
- Full history could be thousands of tokens → expensive LLM calls
- Most of the history is repetitive or irrelevant
- A concise 5-bullet summary captures what matters

The flow:
  Session start: inject summary as a compact system prompt
  Session end:   LLM reads the session, extracts key bullets, merges
                 with existing long-term summary, saves back to Redis

Storage format in Redis:
  Key:   "user:<user_id>:longterm"
  Value: "• User is studying transformer architectures\n• User asked about..."

FIX APPLIED:
  ✅ Lazy Redis initialisation (same reason as short_term.py)
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from upstash_redis import Redis

from config import (
    FAST_MODEL,
    GROQ_API_KEY,
    MEMORY_LONGTERM_TTL,
    UPSTASH_REDIS_TOKEN,
    UPSTASH_REDIS_URL,
)
from memory.short_term import load_history

logger = logging.getLogger(__name__)

# Lazy singleton — same pattern as short_term.py
_redis_client: Optional[Redis] = None


def _get_redis() -> Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis(
            url   = UPSTASH_REDIS_URL,
            token = UPSTASH_REDIS_TOKEN,
        )
    return _redis_client


def _longterm_key(user_id: str) -> str:
    return f"user:{user_id}:longterm"


def load_long_term_summary(user_id: str) -> str:
    """
    Load the long-term summary for a user.
    Returns empty string if no history exists (new user).
    """
    raw = _get_redis().get(_longterm_key(user_id))
    return raw if raw else ""


def summarize_and_save(user_id: str, session_id: str) -> None:
    """
    At the end of a session, use the LLM to summarise what was discussed
    and merge it with the user's existing long-term memory.

    This is called:
    - When the user types 'exit'
    - On process exit via atexit (crash-safe)

    The summarisation is idempotent — if session history is empty, nothing happens.
    """
    history = load_history(session_id)
    if not history:
        logger.info("Session history is empty — nothing to summarise.")
        return

    # Format the session as a readable transcript for the LLM
    transcript = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in history
    )

    existing_summary = load_long_term_summary(user_id)

    # Use the fast model — this is a simple extraction task, not reasoning
    llm = ChatGroq(model=FAST_MODEL, api_key=GROQ_API_KEY, max_tokens=512)

    try:
        response = llm.invoke([
            SystemMessage(content=(
                "You are a memory manager for a RAG chatbot. "
                "Your job is to extract key facts from a conversation transcript "
                "that would be useful to remember in future sessions.\n\n"
                "Rules:\n"
                "1. Output 3-7 concise bullet points starting with •\n"
                "2. Focus on: topics discussed, user's knowledge level, open questions\n"
                "3. Do NOT repeat bullets already in the existing summary\n"
                "4. Be specific — 'user asked about multi-head attention' not 'user asked about model'"
            )),
            HumanMessage(content=(
                f"EXISTING SUMMARY (do not repeat these):\n"
                f"{existing_summary or '(none — this is the first session)'}\n\n"
                f"SESSION TRANSCRIPT:\n{transcript}\n\n"
                f"Output the updated memory bullets:"
            )),
        ])

        new_summary = response.content.strip()
        _get_redis().set(
            _longterm_key(user_id),
            new_summary,
            ex = MEMORY_LONGTERM_TTL,
        )
        logger.info(f"Long-term memory updated for user '{user_id}'.")

    except Exception as error:
        # Don't crash the app if memory saving fails — just log it
        logger.warning(f"Failed to save long-term memory: {error}")


def build_memory_prompt(user_id: str) -> str:
    """
    Build the compact memory string to inject at the start of every session.

    Returns empty string if the user has no prior history.
    The returned string is prepended to the LLM's system prompt.
    """
    summary = load_long_term_summary(user_id)
    if not summary:
        return ""

    return (
        "## What I remember from our previous sessions\n"
        f"{summary}\n"
        "---\n"
    )