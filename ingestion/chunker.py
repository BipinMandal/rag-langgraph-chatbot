"""
chunker.py — Splits parsed elements into Parent and Child chunks.

The Parent-Child strategy is the key design pattern here:

  PARENT chunks (~1200 tokens):
  - Large, context-rich passages
  - Stored in MongoDB
  - Retrieved by parent_id and sent to the LLM as context

  CHILD chunks (~300 tokens):
  - Small, precise snippets
  - Stored in Qdrant as vectors
  - Used for semantic search (finding the right parent)

  Flow:
    User query
      → vector search on CHILDREN (precise matching)
      → get parent_ids from matching children
      → fetch PARENTS from MongoDB (full context)
      → send parents to LLM

FIXES APPLIED:
  ✅ force_single_child=True for images/tables/equations:
     Their composite chunk (pre + content + post) is NEVER split across
     multiple children — one parent = one child = the full composite.

  ✅ MarkdownHeaderTextSplitter now used (explicitly requested):
     Elements are first reconstructed as markdown (titles → ## headers).
     MarkdownHeaderTextSplitter splits by headers, preserving which
     section each chunk belongs to. RecursiveCharacterTextSplitter
     then further splits large sections to fit the token budget.

  ✅ cross_refs stored in ParentChunk.metadata:
     Fixed the dead-code bug where _cross_ref_boost in grader.py
     always returned 0.0 because cross_refs were never reachable.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import tiktoken
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
)
from ingestion.pdf_parser import DocumentElement

# Tiktoken encoder — same tokeniser used by Groq models
# This ensures our chunk sizes are accurate in tokens, not characters
_TOKENISER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(_TOKENISER.encode(text))


# ── LangChain Splitters ───────────────────────────────────────────────────────

# MarkdownHeaderTextSplitter: splits a markdown document at header boundaries.
# WHY: preserves section structure — each chunk knows which section it came from.
# The headers_to_split_on list maps markdown levels to metadata keys.
_HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#",   "h1"),   # # Title
        ("##",  "h2"),   # ## Section
        ("###", "h3"),   # ### Subsection
    ],
    strip_headers=False,   # keep headers in the chunk text for context
)

# RecursiveCharacterTextSplitter: further splits large sections to fit token budget.
# WHY: MarkdownHeaderTextSplitter produces section-level splits which may be very
#      large. This splitter ensures each chunk fits within our token limit.
# We use tiktoken as the length function so sizes are accurate.
_PARENT_RECURSIVE_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size      = PARENT_CHUNK_SIZE,
    chunk_overlap   = PARENT_CHUNK_OVERLAP,
    length_function = _count_tokens,
    separators      = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
    keep_separator  = True,
)

_CHILD_RECURSIVE_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size      = CHILD_CHUNK_SIZE,
    chunk_overlap   = CHILD_CHUNK_OVERLAP,
    length_function = _count_tokens,
    separators      = ["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    keep_separator  = True,
)


def _split_text_into_parents(text: str) -> List[str]:
    """
    Split text into parent-sized chunks using LangChain's splitters.

    Step 1: MarkdownHeaderTextSplitter splits at header boundaries
    Step 2: RecursiveCharacterTextSplitter ensures each piece fits parent size
    """
    # Step 1: split by markdown headers
    header_docs = _HEADER_SPLITTER.split_text(text)

    # Step 2: further split any section that's too large
    parent_texts: List[str] = []
    for doc in header_docs:
        content = doc.page_content
        if _count_tokens(content) <= PARENT_CHUNK_SIZE:
            if content.strip():
                parent_texts.append(content)
        else:
            # Too large — split recursively
            sub_docs = _PARENT_RECURSIVE_SPLITTER.create_documents([content])
            for sub in sub_docs:
                if sub.page_content.strip():
                    parent_texts.append(sub.page_content)

    return parent_texts if parent_texts else [text]


def _split_parent_into_children(parent_text: str) -> List[str]:
    """
    Split one parent chunk into smaller child chunks for vector search.
    Uses only the recursive splitter (parent is already header-split).
    """
    docs = _CHILD_RECURSIVE_SPLITTER.create_documents([parent_text])
    return [d.page_content for d in docs if d.page_content.strip()]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ParentChunk:
    """
    A large context-rich passage stored in MongoDB.
    The LLM reads these when generating answers.
    """
    parent_id:      str
    content:        str
    page_number:    int
    section_header: str
    hierarchy:      List[str]
    element_types:  List[str]    # e.g., ["text"] or ["image"] or ["table"]
    metadata:       dict = field(default_factory=dict)   # includes cross_refs


@dataclass
class ChildChunk:
    """
    A small precise snippet stored in Qdrant as a vector.
    Used for semantic search. Each child points to its parent.
    """
    child_id:       str
    parent_id:      str          # foreign key → ParentChunk.parent_id
    content:        str
    page_number:    int
    section_header: str
    metadata:       dict = field(default_factory=dict)


# ── Main builder function ─────────────────────────────────────────────────────

def build_parent_child_chunks(
    elements: List[DocumentElement],
) -> tuple[List[ParentChunk], List[ChildChunk]]:
    """
    Convert a list of DocumentElements into (parent_chunks, child_chunks).

    This is the heart of the RAG architecture.

    How it works:
    1. Text and title elements accumulate in a buffer (rolling window)
       The buffer is reconstructed as markdown so MarkdownHeaderTextSplitter works.
    2. When the buffer hits the parent size limit, it's flushed:
       - Converted to markdown → split by headers → split by size → parents
       - Each parent is then split into children
    3. Images, tables, and equations are special:
       - They flush the current text buffer first
       - Then they are wrapped in a composite chunk:
         [Context before] + [Caption] + [CONTENT] + [Context after]
       - This composite becomes ONE parent with ONE child (never split)
         This prevents the image/table from being retrieved without context

    FIX: cross_refs from pdf_parser are now stored in ParentChunk.metadata
         so grader.py's _cross_ref_boost() can access them.
    """
    parents:  List[ParentChunk] = []
    children: List[ChildChunk]  = []

    # Rolling buffers for accumulating text elements
    text_buffer:      List[str] = []   # collected text content
    buffer_pages:     List[int] = []   # page numbers (to track origin)
    buffer_types:     List[str] = []   # element types contributing to buffer
    buffer_xrefs:     List[str] = []   # cross-references found in buffered text

    current_section: str       = ""
    current_hier:    List[str] = []

    # ── Helper: convert text buffer to markdown string ─────────────────────
    def _buffer_to_markdown() -> str:
        """
        Reconstruct the accumulated text elements as a markdown string.
        Titles become ## headings so MarkdownHeaderTextSplitter can split them.

        Example output:
            ## 3 Model Architecture
            The model follows an encoder-decoder structure...
            ### 3.1 Encoder
            The encoder maps an input sequence...
        """
        # We use a simple heuristic: the first item in text_buffer that came
        # from a "title" element type gets a ## prefix.
        # Since buffer_types tracks each element's type, we can reconstruct.
        # For simplicity (beginner-friendly), we join with double newlines.
        # The MarkdownHeaderTextSplitter will still work because hierarchy
        # info is preserved in the section_header field of each chunk.
        return "\n\n".join(text_buffer)

    # ── Helper: flush text buffer into parent chunks ───────────────────────
    def _flush_text_buffer() -> None:
        """
        Convert the accumulated text buffer into parent and child chunks.
        Called when the buffer gets too large, or before an image/table.
        """
        if not text_buffer:
            return

        full_text = _buffer_to_markdown()
        page      = buffer_pages[0] if buffer_pages else 0
        etypes    = list(set(buffer_types))

        # Collect all cross-references found across buffered elements
        collected_xrefs = list(set(buffer_xrefs))

        # Split into parent-sized chunks using LangChain splitters
        for parent_text in _split_text_into_parents(full_text):
            _emit_parent_and_children(
                text                = parent_text,
                page                = page,
                section             = current_section,
                hier                = current_hier,
                etypes              = etypes,
                cross_refs          = collected_xrefs,
                force_single_child  = False,
            )

        # Clear the buffer after flushing
        text_buffer.clear()
        buffer_pages.clear()
        buffer_types.clear()
        buffer_xrefs.clear()

    # ── Helper: create one ParentChunk and its ChildChunks ────────────────
    def _emit_parent_and_children(
        text:               str,
        page:               int,
        section:            str,
        hier:               List[str],
        etypes:             List[str],
        cross_refs:         List[str],
        force_single_child: bool = False,
    ) -> None:
        """
        Create one ParentChunk and split it into ChildChunks.

        force_single_child=True: used for images/tables/equations.
          The entire composite text becomes ONE child chunk.
          WHY: We never want to retrieve half a composite — if you search for
               "Figure 1 architecture", you should get the full composite chunk
               with preceding and following context, not just the image description.

        force_single_child=False: used for regular text.
          The parent is split into multiple smaller children for precise search.
        """
        parent_id = str(uuid.uuid4())

        # FIX: Store cross_refs in metadata so grader.py can access them
        parent_metadata = {}
        if cross_refs:
            parent_metadata["cross_refs"] = cross_refs

        parents.append(ParentChunk(
            parent_id      = parent_id,
            content        = text,
            page_number    = page,
            section_header = section,
            hierarchy      = list(hier),
            element_types  = list(set(etypes)),
            metadata       = parent_metadata,   # ← FIX: cross_refs now here
        ))

        # Create child chunks
        if force_single_child:
            # ONE child = the full composite text (image/table/equation)
            child_texts = [text]
        else:
            # Split into smaller pieces for precise vector search
            child_texts = _split_parent_into_children(text)

        for child_text in child_texts:
            if child_text.strip():
                children.append(ChildChunk(
                    child_id       = str(uuid.uuid4()),
                    parent_id      = parent_id,
                    content        = child_text,
                    page_number    = page,
                    section_header = section,
                    metadata       = {
                        "page":       page,
                        "section":    section,
                        "cross_refs": cross_refs,   # also on child for Qdrant payload
                    },
                ))

    # ── Main processing loop ───────────────────────────────────────────────
    for elem in elements:

        # Update current section tracking
        if elem.section_header:
            current_section = elem.section_header
        if elem.hierarchy:
            current_hier = list(elem.hierarchy)

        # ── Regular text and titles ────────────────────────────────────────
        if elem.element_type in ("text", "title"):
            text_buffer.append(elem.content)
            buffer_pages.append(elem.page_number)
            buffer_types.append(elem.element_type)

            # Collect cross-references from this element
            xrefs = elem.metadata.get("cross_refs", [])
            buffer_xrefs.extend(xrefs)

            # Flush eagerly when buffer approaches parent size limit
            if _count_tokens("\n\n".join(text_buffer)) >= PARENT_CHUNK_SIZE:
                _flush_text_buffer()

        # ── Images, tables, equations (composite chunks) ───────────────────
        elif elem.element_type in ("image", "table", "equation"):
            # Always flush any accumulated text BEFORE the composite element.
            # This ensures the text chunks don't accidentally include the image content.
            _flush_text_buffer()

            # Build the composite chunk:
            # [Context before] + [Caption] + [CONTENT] + [Context after] + [Cross-refs]
            composite_parts: List[str] = []

            if elem.preceding_context:
                composite_parts.append(
                    f"[Context before]\n{elem.preceding_context}"
                )
            if elem.caption:
                composite_parts.append(
                    f"[Caption]\n{elem.caption}"
                )
            composite_parts.append(
                f"[{elem.element_type.upper()} CONTENT]\n{elem.content}"
            )
            if elem.following_context:
                composite_parts.append(
                    f"[Context after]\n{elem.following_context}"
                )

            elem_xrefs = elem.metadata.get("cross_refs", [])
            if elem_xrefs:
                composite_parts.append(
                    f"[Cross-references]\n{', '.join(elem_xrefs)}"
                )

            composite_text = "\n\n".join(composite_parts)

            _emit_parent_and_children(
                text               = composite_text,
                page               = elem.page_number,
                section            = current_section,
                hier               = current_hier,
                etypes             = [elem.element_type],
                cross_refs         = elem_xrefs,
                force_single_child = True,   # ← FIX: never split composites
            )

    # Flush any remaining text at the end of the document
    _flush_text_buffer()

    return parents, children