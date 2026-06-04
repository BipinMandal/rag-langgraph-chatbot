"""
ingest_pipeline.py — One-time pipeline to process a PDF and populate the databases.

Run this ONCE before starting the chatbot.
It is safe to run multiple times — all operations are idempotent (upserts).

What it does:
  Step 1: Parse the PDF into structured elements (text, tables, images, equations)
  Step 2: Caption images using Groq vision LLM
  Step 3: Split elements into parent + child chunks
  Step 4: Store parent chunks in MongoDB Atlas
  Step 5: Embed + index child chunks in Qdrant Cloud

Usage:
  uv run ingest --pdf attention_is_all_you_need.pdf
  # or
  uv run python ingest_pipeline.py --pdf attention_is_all_you_need.pdf

FIX APPLIED:
  ✅ Placeholder-content children are filtered out before indexing.
     Images that failed captioning get content like "[IMAGE — no description available]".
     Storing these as vectors pollutes the vector index with meaningless embeddings.
     We skip them so they don't appear in search results.
"""
from __future__ import annotations

import argparse
import logging
import sys

# Configure logging BEFORE importing other modules
# This ensures log messages from all modules are visible during ingestion
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
    stream  = sys.stdout,
)

from document_store.mongodb_store  import MongoParentStore
from ingestion.chunker             import build_parent_child_chunks
from ingestion.image_captioner     import PLACEHOLDER_CONTENT, caption_images
from ingestion.pdf_parser          import PDFParser
from vectorstore.qdrant_store      import QdrantHybridStore

logger = logging.getLogger(__name__)


def run_ingestion(pdf_path: str) -> None:
    """
    Full ingestion pipeline for one PDF file.
    Populates MongoDB (parent chunks) and Qdrant (child chunk vectors).
    """
    logger.info("=" * 60)
    logger.info("  RAG Ingestion Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Parse PDF ──────────────────────────────────────────────────
    logger.info("\n[Step 1/5] Parsing PDF …")
    parser   = PDFParser(pdf_path=pdf_path, image_dir="extracted_images")
    elements = parser.parse()

    # Show a breakdown of what was found
    type_counts = {}
    for e in elements:
        type_counts[e.element_type] = type_counts.get(e.element_type, 0) + 1
    logger.info(f"  Element breakdown: {type_counts}")

    # ── Step 2: Caption images with Groq vision ────────────────────────────
    logger.info("\n[Step 2/5] Captioning images with Groq vision model …")
    logger.info("  (Each image requires one API call — be patient)")
    elements = caption_images(elements)

    # Show captioning results
    captioned = sum(
        1 for e in elements
        if e.element_type == "image" and e.content not in PLACEHOLDER_CONTENT
    )
    failed = sum(
        1 for e in elements
        if e.element_type == "image" and e.content in PLACEHOLDER_CONTENT
    )
    logger.info(f"  Captioned: {captioned}  |  Failed/skipped: {failed}")

    # ── Step 3: Build parent-child chunks ─────────────────────────────────
    logger.info("\n[Step 3/5] Building parent-child chunks …")
    parents, children = build_parent_child_chunks(elements)
    logger.info(f"  Created {len(parents)} parent chunks and {len(children)} child chunks.")

    # Show element type distribution across parents
    parent_type_counts = {}
    for p in parents:
        for t in p.element_types:
            parent_type_counts[t] = parent_type_counts.get(t, 0) + 1
    logger.info(f"  Parent types: {parent_type_counts}")

    # ── Step 4: Store parent chunks in MongoDB ────────────────────────────
    logger.info("\n[Step 4/5] Storing parent chunks in MongoDB Atlas …")
    mongo = MongoParentStore()
    parent_dicts = [
        {
            "parent_id":      p.parent_id,
            "content":        p.content,
            "page_number":    p.page_number,
            "section_header": p.section_header,
            "hierarchy":      p.hierarchy,
            "element_types":  p.element_types,
            # FIX: metadata includes cross_refs (needed by grader.py)
            "metadata":       p.metadata,
        }
        for p in parents
    ]
    mongo.upsert_parents(parent_dicts)
    logger.info(f"  MongoDB total: {mongo.count()} parent docs.")
    mongo.close()

    # ── Step 5: Index child chunks in Qdrant ──────────────────────────────
    logger.info("\n[Step 5/5] Indexing child chunks in Qdrant Cloud …")
    logger.info("  (Generating embeddings locally — first run downloads ~90 MB of models)")

    # FIX: Filter out children whose content is a placeholder.
    # These come from images that failed captioning.
    # Storing "[IMAGE — no description available]" as a vector wastes index space
    # and can mislead search results.
    valid_children = [
        c for c in children
        if c.content.strip() not in PLACEHOLDER_CONTENT
    ]
    skipped = len(children) - len(valid_children)
    if skipped > 0:
        logger.info(f"  Skipping {skipped} children with placeholder content.")

    child_dicts = [
        {
            "child_id":       c.child_id,
            "parent_id":      c.parent_id,
            "content":        c.content,
            "page_number":    c.page_number,
            "section_header": c.section_header,
            # Pass metadata so Qdrant payload includes cross_refs
            "metadata":       c.metadata,
        }
        for c in valid_children
    ]
    qdrant = QdrantHybridStore()
    qdrant.upsert_children(child_dicts)
    logger.info(f"  Qdrant total: {qdrant.count()} child chunks.")

    # ── Done ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  ✅  Ingestion complete!")
    logger.info(f"      Parents in MongoDB : {len(parent_dicts)}")
    logger.info(f"      Children in Qdrant : {len(valid_children)}")
    logger.info("      You can now run: uv run chat --user your_name")
    logger.info("=" * 60)


def main() -> None:
    """Entry point for `uv run ingest` (defined in pyproject.toml scripts)."""
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into the RAG chatbot knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run ingest --pdf attention_is_all_you_need.pdf
  uv run python ingest_pipeline.py --pdf my_paper.pdf

Download the test PDF:
  curl -L "https://arxiv.org/pdf/1706.03762.pdf" -o attention_is_all_you_need.pdf
        """
    )
    parser.add_argument(
        "--pdf",
        required    = True,
        metavar     = "PATH",
        help        = "Path to the PDF file to ingest (e.g., attention_is_all_you_need.pdf)",
    )
    args = parser.parse_args()
    run_ingestion(args.pdf)


if __name__ == "__main__":
    main()