"""
short_term.py — In-session chat history stored in Upstash Redis.

What is short-term memory?
Within a single conversation session, the chatbot needs to remember
what was said earlier. Without this, every question is treated in isolation
and the chatbot can't answer follow-ups like "explain the second point again".

How it works:
- Each session has a unique session_id (UUID)
- Chat turns are stored as a JSON list in Redis
- The list expires after MEMORY_SESSION_TTL (1 hour of inactivity)
- On each Q&A turn, we append to the list and refresh the TTL

Storage format in Redis:
  Key:   "session:<session_id>:history"
  Value: '[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]'

FIX APPLIED:
  ✅ Lazy Redis initialisation — the Redis client is created on first use,
     not at module import time. This prevents a startup crash if
     UPSTASH_REDIS_URL or UPSTASH_REDIS_TOKEN are temporarily unavailable.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from upstash_redis import Redis

from config import MEMORY_SESSION_TTL, UPSTASH_REDIS_TOKEN, UPSTASH_REDIS_URL

logger = logging.getLogger(__name__)

# Lazy singleton — created on first use, not at import time
_redis_client: Optional[Redis] = None


def _get_redis() -> Redis:
    """
    Get (or create) the Redis client.

    WHY lazy init?
    If Redis credentials are misconfigured, we want the error to appear
    when the user actually tries to use memory — not during import.
    This gives a much clearer error message.
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis(
            url   = UPSTASH_REDIS_URL,
            token = UPSTASH_REDIS_TOKEN,
        )
    return _redis_client


def _session_key(session_id: str) -> str:
    """Build the Redis key for a session's history."""
    return f"session:{session_id}:history"


def load_history(session_id: str) -> List[Dict]:
    """
    Load the chat history for a session.
    Returns an empty list if the session doesn't exist or has expired.
    """
    raw = _get_redis().get(_session_key(session_id))
    if not raw:
        return []
    return json.loads(raw)


def append_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """
    Append one complete Q&A turn to the session history.
    Refreshes the TTL so the session stays alive as long as the user is active.
    """
    history = load_history(session_id)
    history.append({"role": "user",      "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})

    _get_redis().set(
        _session_key(session_id),
        json.dumps(history),
        ex = MEMORY_SESSION_TTL,   # TTL reset on every new turn
    )


def clear_history(session_id: str) -> None:
    """Delete the session history (e.g., on explicit logout)."""
    _get_redis().delete(_session_key(session_id))