"""
mongodb_store.py — Stores and retrieves parent chunks from MongoDB Atlas.

Why MongoDB?
- Free M0 tier (512 MB) is enough for most research papers
- Simple document store — we store dicts, we retrieve dicts
- Fast lookup by parent_id (with an index)

How it's used in the pipeline:
  1. During ingestion: store all parent chunks
  2. During retrieval: given a list of parent_ids from Qdrant,
     fetch the full parent documents for the LLM
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from pymongo import MongoClient, UpdateOne

from config import MONGODB_DB, MONGODB_PARENT_COLL, MONGODB_URI

logger = logging.getLogger(__name__)


class MongoParentStore:
    """
    A simple wrapper around a MongoDB collection for parent chunks.

    All operations are idempotent — safe to re-run ingestion
    without creating duplicates (upsert on parent_id).
    """

    def __init__(self) -> None:
        # Connect to MongoDB Atlas
        # The MongoClient handles connection pooling automatically
        self._client = MongoClient(MONGODB_URI)
        self._col    = self._client[MONGODB_DB][MONGODB_PARENT_COLL]

        # Create an index on parent_id for fast O(1) lookups.
        # background=True means indexing doesn't block the app.
        self._col.create_index("parent_id", unique=True, background=True)
        logger.info("MongoDB connected and index ensured.")

    # ── Write operations ──────────────────────────────────────────────────

    def upsert_parents(self, parents: List[dict]) -> None:
        """
        Store a list of parent chunk dicts.
        If a parent with the same parent_id already exists, it is updated.
        This means you can safely re-run ingestion without creating duplicates.
        """
        if not parents:
            return

        # bulk_write sends all operations in one network round-trip (fast)
        operations = [
            UpdateOne(
                filter  = {"parent_id": p["parent_id"]},   # find by this
                update  = {"$set": p},                      # replace with this
                upsert  = True,                             # insert if not found
            )
            for p in parents
        ]
        result = self._col.bulk_write(operations)
        logger.info(
            f"  MongoDB: {result.upserted_count} inserted, "
            f"{result.modified_count} updated."
        )

    # ── Read operations ───────────────────────────────────────────────────

    def get_parents_by_ids(self, parent_ids: List[str]) -> List[Dict]:
        """
        Fetch multiple parent documents by their IDs in one query.
        The {"_id": 0} projection excludes MongoDB's internal _id field.
        """
        docs = list(
            self._col.find(
                {"parent_id": {"$in": parent_ids}},
                {"_id": 0}   # don't return MongoDB's internal _id
            )
        )
        logger.debug(f"Fetched {len(docs)}/{len(parent_ids)} parent docs.")
        return docs

    def count(self) -> int:
        """Return total number of stored parent chunks."""
        return self._col.count_documents({})

    def close(self) -> None:
        """Close the MongoDB connection."""
        self._client.close()