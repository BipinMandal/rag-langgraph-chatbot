"""
qdrant_store.py — Manages the Qdrant vector database for child chunks.

What is a vector database?
A regular database searches by exact match (find where name="Alice").
A vector database searches by MEANING — it finds chunks that are
semantically similar to the query, even with different words.

Why HYBRID search (dense + sparse)?
- Dense (semantic): finds conceptually similar content
  Example: "multi-head attention" matches "parallel attention heads"
- Sparse (BM25/keyword): finds exact word matches
  Example: "softmax" must contain "softmax", not just "activation function"

Combining both gives much better results than either alone.
The RRF (Reciprocal Rank Fusion) algorithm merges the two ranked lists.

FIXES APPLIED:
  ✅ Retry logic on batch upsert — network hiccups won't fail the whole ingestion
  ✅ Beginner-friendly error messages on connection failure
"""
from __future__ import annotations

import logging
import time
from typing import List

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from config import QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL

logger = logging.getLogger(__name__)

# Model names for embeddings
_DENSE_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dimensional
_SPARSE_MODEL = "Qdrant/bm25"                               # BM25 sparse
_DENSE_DIM    = 384   # must match the model output dimension

# Batch size for upserting — stays within Qdrant's API limits
_UPSERT_BATCH_SIZE = 100

# Retry settings for transient network errors
_MAX_RETRIES    = 3
_RETRY_DELAY_S  = 2   # seconds between retries


class QdrantHybridStore:
    """
    Manages a Qdrant collection with both dense and sparse vectors.

    Each point in the collection = one child chunk with:
    - dense vector:  384-float semantic embedding
    - sparse vector: BM25 keyword indices+weights
    - payload:       metadata (parent_id, content, page, section)
    """

    def __init__(self) -> None:
        logger.info("Connecting to Qdrant Cloud …")
        self._client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        # Load embedding models locally (downloaded on first use, then cached)
        logger.info("Loading dense embedding model (all-MiniLM-L6-v2) …")
        logger.info("  (First run downloads ~90 MB — subsequent runs are instant)")
        self._dense_model  = TextEmbedding(model_name=_DENSE_MODEL)

        logger.info("Loading sparse BM25 model …")
        self._sparse_model = SparseTextEmbedding(model_name=_SPARSE_MODEL)

        self._ensure_collection()
        logger.info("QdrantHybridStore ready ✓")

    # ── Collection setup ──────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """
        Create the Qdrant collection if it doesn't exist yet.
        Safe to call multiple times — checks before creating.
        """
        existing_names = [c.name for c in self._client.get_collections().collections]

        if QDRANT_COLLECTION in existing_names:
            logger.info(f"Collection '{QDRANT_COLLECTION}' already exists — skipping creation.")
            return

        logger.info(f"Creating collection '{QDRANT_COLLECTION}' …")
        self._client.create_collection(
            collection_name       = QDRANT_COLLECTION,
            vectors_config        = {
                # Dense vector config: 384-dim, cosine similarity
                "dense": VectorParams(size=_DENSE_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config = {
                # Sparse vector config: BM25 indices, stored in memory for speed
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                ),
            },
        )

        # Create a payload index on section_header for fast metadata filtering.
        # KEYWORD type = exact string match (fast).
        self._client.create_payload_index(
            collection_name = QDRANT_COLLECTION,
            field_name      = "section_header",
            field_schema    = PayloadSchemaType.KEYWORD,
        )
        logger.info(f"Collection '{QDRANT_COLLECTION}' created with indexes ✓")

    # ── Embedding helpers ─────────────────────────────────────────────────

    def _embed_dense(self, texts: List[str]) -> List[List[float]]:
        """Convert a list of texts to dense (semantic) vectors."""
        return [list(vec) for vec in self._dense_model.embed(texts)]

    def _embed_sparse(self, texts: List[str]) -> List[SparseVector]:
        """Convert a list of texts to sparse (BM25 keyword) vectors."""
        return [
            SparseVector(
                indices=sparse.indices.tolist(),
                values=sparse.values.tolist(),
            )
            for sparse in self._sparse_model.embed(texts)
        ]

    # ── Indexing (write) ──────────────────────────────────────────────────

    def upsert_children(self, child_dicts: List[dict]) -> None:
        """
        Index a list of child chunk dicts into Qdrant.

        Each dict needs: child_id, parent_id, content, page_number, section_header

        FIX: Retry logic added — if a batch fails due to a network error,
        we retry up to _MAX_RETRIES times before giving up.
        """
        if not child_dicts:
            return

        logger.info(f"Embedding and indexing {len(child_dicts)} child chunks …")

        # Generate embeddings for all texts at once (more efficient than one-by-one)
        texts       = [c["content"] for c in child_dicts]
        dense_vecs  = self._embed_dense(texts)
        sparse_vecs = self._embed_sparse(texts)

        # Build Qdrant PointStruct objects (each = one document in the collection)
        points = [
            PointStruct(
                id     = c["child_id"],   # UUID string
                vector = {
                    "dense":  dense_vecs[i],
                    "sparse": sparse_vecs[i],
                },
                payload = {
                    "parent_id":      c["parent_id"],
                    "content":        c["content"],
                    "page_number":    c.get("page_number", 0),
                    "section_header": c.get("section_header", ""),
                    "cross_refs":     c.get("metadata", {}).get("cross_refs", []),
                },
            )
            for i, c in enumerate(child_dicts)
        ]

        # Upsert in batches of 100 with retry on failure
        total_batches = (len(points) + _UPSERT_BATCH_SIZE - 1) // _UPSERT_BATCH_SIZE
        for batch_num, start in enumerate(range(0, len(points), _UPSERT_BATCH_SIZE), 1):
            batch = points[start : start + _UPSERT_BATCH_SIZE]

            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    self._client.upsert(
                        collection_name = QDRANT_COLLECTION,
                        points          = batch,
                    )
                    logger.info(f"  Batch {batch_num}/{total_batches} indexed ✓")
                    break   # success — exit retry loop

                except Exception as error:
                    if attempt < _MAX_RETRIES:
                        logger.warning(
                            f"  Batch {batch_num} failed (attempt {attempt}/{_MAX_RETRIES}): "
                            f"{error}. Retrying in {_RETRY_DELAY_S}s …"
                        )
                        time.sleep(_RETRY_DELAY_S)
                    else:
                        logger.error(
                            f"  Batch {batch_num} failed after {_MAX_RETRIES} attempts. "
                            f"Skipping this batch. Error: {error}"
                        )

        logger.info(f"Indexing complete — {len(points)} chunks in Qdrant ✓")

    # ── Search (read) ──────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query:          str,
        top_k:          int = 10,
        section_filter: str | None = None,
    ) -> List[dict]:
        """
        Search for the most relevant child chunks using HYBRID search.

        How it works:
        1. Embed the query with both dense and sparse models
        2. Run two parallel searches:
           - Dense search: finds semantically similar content
           - Sparse search: finds exact keyword matches
        3. Merge results with RRF (Reciprocal Rank Fusion):
           RRF gives a higher score to documents that rank well in BOTH lists
        4. Return the top_k combined results

        section_filter: optional — restrict search to a specific paper section
        """
        # Embed the query
        dense_vec  = self._embed_dense([query])[0]
        sparse_vec = self._embed_sparse([query])[0]

        # Build optional metadata filter
        metadata_filter = None
        if section_filter:
            metadata_filter = Filter(must=[
                FieldCondition(
                    key   = "section_header",
                    match = MatchValue(value=section_filter),
                )
            ])

        # Run hybrid search using Qdrant's Prefetch + FusionQuery
        results = self._client.query_points(
            collection_name = QDRANT_COLLECTION,
            prefetch        = [
                # Prefetch more candidates than needed (top_k * 2)
                # so RRF has enough items to rank from both sources
                Prefetch(
                    query  = dense_vec,
                    using  = "dense",
                    limit  = top_k * 2,
                    filter = metadata_filter,
                ),
                Prefetch(
                    query  = sparse_vec,
                    using  = "sparse",
                    limit  = top_k * 2,
                    filter = metadata_filter,
                ),
            ],
            # FusionQuery with RRF merges both ranked lists
            query = FusionQuery(fusion=Fusion.RRF),
            limit = top_k,
        )

        # Convert Qdrant result objects to plain dicts
        hits = []
        for point in results.points:
            payload              = dict(point.payload)
            payload["score"]     = point.score
            payload["child_id"]  = point.id
            hits.append(payload)

        return hits

    def count(self) -> int:
        """Return total number of indexed child chunks."""
        return self._client.get_collection(QDRANT_COLLECTION).points_count or 0