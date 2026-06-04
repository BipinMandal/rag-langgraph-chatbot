"""
main.py — Interactive CLI chatbot.

Starts a conversation loop where you can ask questions about the
"Attention Is All You Need" paper. The chatbot uses the full
LangGraph RAG pipeline for every question.

Usage:
  uv run chat --user alice
  # or
  uv run python main.py --user alice

Type 'exit', 'quit', or press Ctrl+C to end the session.
Long-term memory is saved automatically when you exit.

FIXES APPLIED:
  ✅ atexit and signal handlers registered INSIDE chat() — not at module level.
     This means they only activate when the chatbot actually starts,
     not when main.py is merely imported (e.g., in tests).
  ✅ _save_memory_once() uses a flag to guarantee it runs exactly once,
     regardless of how many exit paths are taken (exit command, Ctrl+C,
     SIGTERM from OS, or normal process end via atexit).
  ✅ Clear, beginner-friendly error messages throughout.
"""
from __future__ import annotations

import atexit
import logging
import signal
import sys
import uuid

# Show only WARNING and above in the console during normal use.
# Change to logging.INFO to see pipeline step-by-step logs (useful for debugging).
logging.basicConfig(
    level   = logging.WARNING,
    format  = "%(levelname)-8s  %(message)s",
    stream  = sys.stdout,
)

from graph.pipeline   import rag_graph
from memory.long_term import summarize_and_save

# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║        RAG Chatbot — Attention Is All You Need (2017)        ║
║   LangGraph · Groq · Qdrant Cloud · MongoDB · Upstash Redis  ║
╠══════════════════════════════════════════════════════════════╣
║  Ask anything about the Transformer paper.                   ║
║  Examples:                                                   ║
║    "How does multi-head attention work?"                     ║
║    "What does Figure 1 show?"                                ║
║    "Compare the model sizes in Table 2"                      ║
║    "Why did the authors use positional encoding?"            ║
╠══════════════════════════════════════════════════════════════╣
║  Type 'exit' or press Ctrl+C to end the session.             ║
╚══════════════════════════════════════════════════════════════╝
"""


# ── Chat function ─────────────────────────────────────────────────────────────

def chat(user_id: str) -> None:
    """
    Main conversation loop.

    Each call to this function is one complete session.
    A unique session_id is generated for every session so short-term
    memory (Redis) doesn't mix up different sessions.
    """
    session_id = str(uuid.uuid4())

    print(BANNER)
    print(f"  User       : {user_id}")
    print(f"  Session ID : {session_id}")
    print(f"  (Debug tip : set logging level to INFO in main.py to see pipeline steps)\n")

    # ── Crash-safe memory saving ───────────────────────────────────────────
    # We want long-term memory to be saved even if the user Ctrl+C's or
    # the process is killed. We use three mechanisms:

    # Guard to ensure we save exactly ONCE no matter how many exit paths fire
    memory_saved = {"done": False}

    def save_memory_once(reason: str = "unknown") -> None:
        """Save long-term memory exactly once. Thread-safe via dict flag."""
        if memory_saved["done"]:
            return
        memory_saved["done"] = True
        print(f"\n  💾  Saving session memory ({reason}) …")
        try:
            summarize_and_save(user_id=user_id, session_id=session_id)
            print("  ✓  Memory saved.")
        except Exception as err:
            print(f"  ⚠️  Memory save failed: {err}")

    # FIX: Register atexit INSIDE chat() — only runs when chatbot is actually used.
    # atexit fires when the Python process exits normally.
    atexit.register(save_memory_once, "process exit")

    # FIX: Register signal handlers INSIDE chat().
    # SIGINT  = Ctrl+C pressed by user
    # SIGTERM = `kill <pid>` or Docker stop, etc.
    def signal_handler(sig, frame) -> None:
        save_memory_once(f"signal {sig}")
        sys.exit(0)

    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ── Main question-answer loop ──────────────────────────────────────────
    while True:
        # Get input — handle EOF gracefully (e.g., piped input)
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        # Skip empty input (user just pressed Enter)
        if not user_input:
            continue

        # Exit commands
        if user_input.lower() in ("exit", "quit", "bye", "q"):
            break

        # ── Build the initial state for this query ─────────────────────────
        # Every field in GraphState must be provided.
        # load_memory node will reset most of these, but we initialise
        # them here so the TypedDict is fully populated.
        initial_state = {
            "query":             user_input,
            "user_id":           user_id,
            "session_id":        session_id,
            # These will all be overwritten by load_memory:
            "chat_history":      [],
            "memory_context":    "",
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

        # ── Run the LangGraph pipeline ─────────────────────────────────────
        try:
            print("  (thinking …)")
            final_state = rag_graph.invoke(initial_state)
            answer      = final_state.get(
                "generation",
                "Sorry, I was unable to generate an answer. Please try again."
            )
            print(f"\nAssistant: {answer}\n")

        except Exception as error:
            # Surface errors clearly rather than crashing silently
            print(f"\n  ❌  Error during processing: {error}")
            print("  Please check your .env file and ensure all services are running.\n")
            # Continue the loop — don't crash the whole session on one error

    # ── Clean exit ─────────────────────────────────────────────────────────
    save_memory_once("user exit")
    print("\n  Goodbye! 👋\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for `uv run chat` (defined in pyproject.toml scripts)."""
    import argparse
    parser = argparse.ArgumentParser(
        description="RAG Chatbot — Ask questions about the Attention Is All You Need paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run chat --user alice
  uv run python main.py --user bob

Before running this, make sure you have ingested the PDF:
  uv run ingest --pdf attention_is_all_you_need.pdf
        """
    )
    parser.add_argument(
        "--user",
        default = "default_user",
        metavar = "NAME",
        help    = "Your username — used to load/save your long-term memory (default: default_user)",
    )
    args = parser.parse_args()
    chat(args.user)


if __name__ == "__main__":
    main()