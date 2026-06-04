"""
config.py — Central configuration for the entire application.

All API keys, model names, and tuning parameters live here.
Import from this file everywhere else — never call os.getenv() in other files.

FIX APPLIED:
  ✅ _require() validates every key at startup.
     If a key is missing, you get a clear error message immediately
     instead of a cryptic crash deep inside LangChain.
"""

import os
from dotenv import load_dotenv

# Load variables from the .env file into os.environ
# This must run before any os.getenv() call
load_dotenv()


# ── Helper: validate required environment variables ───────────────────────────
def _require(name: str) -> str:
    """
    Read an environment variable. Crash immediately with a helpful message
    if it is missing or empty.

    Why this matters: Without this, a missing API key causes a confusing
    AttributeError or ConnectionRefusedError deep inside a library —
    very hard to debug for beginners.
    """
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"\n\n❌  Required environment variable '{name}' is not set.\n"
            f"    Steps to fix:\n"
            f"    1. Copy .env.example to .env  →  cp .env.example .env\n"
            f"    2. Open .env and fill in your '{name}' value.\n"
            f"    3. Save the file and run again.\n"
        )
    return value


# ── API Keys (validated at startup) ──────────────────────────────────────────
GROQ_API_KEY        = _require("GROQ_API_KEY")
TAVILY_API_KEY      = _require("TAVILY_API_KEY")
QDRANT_URL          = _require("QDRANT_URL")
QDRANT_API_KEY      = _require("QDRANT_API_KEY")
MONGODB_URI         = _require("MONGODB_URI")
UPSTASH_REDIS_URL   = _require("UPSTASH_REDIS_URL")
UPSTASH_REDIS_TOKEN = _require("UPSTASH_REDIS_TOKEN")


# ── Groq Model Names ──────────────────────────────────────────────────────────
# Each model is chosen for a specific job:

# Best quality — used for generating the final answer
# 70 billion parameters → deeper reasoning, better answers
GENERATION_MODEL = "llama-3.3-70b-versatile"

# Fastest — used for quick classification tasks (routing, grading, rewriting)
# 8 billion parameters → very fast, 30,000 tokens/min free limit
FAST_MODEL = "llama-3.1-8b-instant"

# Vision — can understand images, used to describe figures from the paper
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


# ── Chunking Parameters ───────────────────────────────────────────────────────
# Parent chunks: larger, context-rich passages stored in MongoDB
# Child  chunks: smaller, precise snippets stored in Qdrant for search

# Why two sizes?
# - Small chunks → better vector search (more focused meaning per chunk)
# - Large chunks → better answers (more context for the LLM)
# Solution: search with small (child), answer with large (parent)

PARENT_CHUNK_SIZE    = 1200   # tokens — roughly 900 words
PARENT_CHUNK_OVERLAP = 150    # tokens shared between adjacent parents (continuity)
CHILD_CHUNK_SIZE     = 300    # tokens — roughly 225 words
CHILD_CHUNK_OVERLAP  = 50     # tokens shared between adjacent children


# ── Retrieval Parameters ──────────────────────────────────────────────────────
TOP_K_CHILD  = 10   # how many child chunks to retrieve from Qdrant
TOP_K_FINAL  = 5    # how many unique parent docs to pass to the LLM (after dedup)

# Relevance threshold: documents scoring below this are filtered out
# 0.0 = accept everything, 1.0 = only exact matches
# 0.6 = "reasonably relevant" — good balance for technical papers
RELEVANCE_THRESHOLD = 0.6

# Minimum number of graded docs needed before we trust the RAG results.
# If fewer docs pass grading, we fall back to web search.
# Set to 1: even a single highly-relevant doc is enough.
MIN_GRADED_DOCS = 1


# ── Storage Names ─────────────────────────────────────────────────────────────
QDRANT_COLLECTION   = "rag_chatbot_children"   # Qdrant collection name
MONGODB_DB          = "rag_chatbot"             # MongoDB database name
MONGODB_PARENT_COLL = "parent_docs"            # MongoDB collection for parent chunks


# ── Memory TTLs (Time-To-Live) ────────────────────────────────────────────────
# Redis automatically deletes keys after these durations.
MEMORY_SESSION_TTL  = 3600               # 1 hour  → short-term chat history
MEMORY_LONGTERM_TTL = 60 * 60 * 24 * 90 # 90 days → long-term user summary


# ── Retry / Safety ────────────────────────────────────────────────────────────
# How many times the validator can reject an answer before we give up retrying
MAX_VALIDATOR_RETRIES = 2