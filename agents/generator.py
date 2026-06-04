"""
generator.py — Generates the final answer and extracts citation metadata.

UPDATED:
  - After generating the answer, parses each graded_doc to extract:
      • element_type (text / image / table / equation)
      • image_path and image_base64 (for rendering images in UI)
      • caption (figure/table captions)
      • table markdown (for rendering tables in UI)
  - Stores all of this in state["cited_sources"] for the UI to render.
"""
from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import GENERATION_MODEL, GROQ_API_KEY
from graph.state import GraphState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert assistant specialised in transformer architectures
and the "Attention Is All You Need" paper (Vaswani et al., 2017).

You will be given numbered context documents. Use ONLY these documents to answer.

RULES:
1. Cite your sources: reference [Doc N], the section name, or the page number
   whenever you use information from a specific document.
   Example: "As described in [Doc 2] (Section 3.2, Page 4), the attention function..."
2. Include table data: if context contains a Markdown table, quote the relevant rows.
3. Reference figures: if context contains an image description, say
   "As shown in [Doc N] / Figure X, ..."
4. Be honest: if the context does not contain enough information to answer,
   say "I don't have enough information in my knowledge base to answer that."
5. Never fabricate: do not invent equations, numbers, or citations not in the context.
6. Be technical: this is a research paper — use precise terminology.
"""

# ── Regex patterns for parsing composite chunk sections ──────────────────────
# These match the section markers written by chunker.py
_SECTION_PATTERNS = {
    "preceding": re.compile(r"\[Context before\]\s*(.*?)(?=\[|$)", re.DOTALL),
    "caption":   re.compile(r"\[Caption\]\s*(.*?)(?=\[|$)",         re.DOTALL),
    "content":   re.compile(
        r"\[(IMAGE|TABLE|EQUATION) CONTENT\]\s*(.*?)(?=\[|$)", re.DOTALL
    ),
    "following": re.compile(r"\[Context after\]\s*(.*?)(?=\[|$)",   re.DOTALL),
}


def _format_context_for_llm(docs: List[Dict]) -> str:
    """Format retrieved documents as numbered context for the LLM."""
    if not docs:
        return "No relevant documents were found."

    parts = []
    for i, doc in enumerate(docs, start=1):
        header_parts = [f"[Doc {i}]"]
        if doc.get("section_header"):
            header_parts.append(f"Section: {doc['section_header']}")
        if doc.get("page_number"):
            header_parts.append(f"Page {doc['page_number']}")
        if doc.get("source"):
            header_parts.append(f"Source: {doc['source']}")
        header  = " | ".join(header_parts)
        content = doc.get("content", "").strip()
        parts.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(parts)


def _detect_element_type(content: str) -> str:
    """
    Detect the element type of a document based on its content markers.
    These markers are written by chunker.py when building composite chunks.
    """
    if "[IMAGE CONTENT]" in content:
        return "image"
    if "[TABLE CONTENT]" in content:
        return "table"
    if "[EQUATION CONTENT]" in content:
        return "equation"
    return "text"


def _extract_caption(content: str) -> str:
    """Extract the [Caption] section from a composite chunk."""
    match = _SECTION_PATTERNS["caption"].search(content)
    return match.group(1).strip() if match else ""


def _extract_table_markdown(content: str) -> str:
    """
    Extract the Markdown table from a composite chunk's [TABLE CONTENT] section.
    Returns empty string if no table is found.
    """
    match = _SECTION_PATTERNS["content"].search(content)
    if match and match.group(1) == "TABLE":
        return match.group(2).strip()
    return ""


def _extract_image_data(content: str, image_dir: str = "extracted_images") -> Dict:
    """
    Try to find the image file corresponding to this composite chunk.

    The caption usually contains the figure number, e.g., "Figure 1: ..."
    We use that to locate the saved image file from the ingestion step.

    Returns a dict with image_path and image_base64 (both may be empty strings
    if the image file cannot be found).
    """
    caption  = _extract_caption(content)
    img_path = ""
    img_b64  = ""

    # Try to find a figure number in the caption: "Figure 1", "Fig. 2", etc.
    fig_match = re.search(r"(figure|fig\.?)\s*(\d+)", caption, re.IGNORECASE)
    if fig_match:
        fig_num   = fig_match.group(2)
        image_dir_path = Path(image_dir)

        # Search for image files that match this figure number
        # The ingestion pipeline saves images as page{N}_img{N}.{ext}
        # We do a loose search across all saved images
        if image_dir_path.exists():
            candidates = list(image_dir_path.glob(f"*.jpeg")) + \
                         list(image_dir_path.glob(f"*.jpg"))  + \
                         list(image_dir_path.glob(f"*.png"))

            # Heuristic: use the image file whose index roughly matches the figure num
            # (e.g., Figure 1 → first image, Figure 2 → second image)
            idx = max(0, int(fig_num) - 1)
            if idx < len(candidates):
                img_path = str(candidates[idx])
                try:
                    raw      = Path(img_path).read_bytes()
                    img_b64  = base64.b64encode(raw).decode("utf-8")
                except Exception:
                    img_b64 = ""

    return {"image_path": img_path, "image_base64": img_b64}


def _build_cited_sources(docs: List[Dict]) -> List[Dict[str, Any]]:
    """
    Build the cited_sources list from graded documents.

    Each entry contains everything the UI needs to render one citation:
    - For text:      the text snippet
    - For images:    caption + description + image file data (path + base64)
    - For tables:    caption + markdown table
    - For equations: the equation text + surrounding context
    - For web:       URL + snippet
    """
    sources: List[Dict[str, Any]] = []

    for i, doc in enumerate(docs, start=1):
        content      = doc.get("content", "")
        element_type = _detect_element_type(content)
        caption      = _extract_caption(content)

        source: Dict[str, Any] = {
            "doc_number":     i,
            "section_header": doc.get("section_header", ""),
            "page_number":    doc.get("page_number", 0),
            "element_type":   element_type,
            "content":        content,
            "caption":        caption,
            "source_url":     doc.get("source", ""),    # web search results only
            "image_path":     "",
            "image_base64":   "",
            "table_markdown": "",
        }

        if element_type == "image":
            img_data               = _extract_image_data(content)
            source["image_path"]   = img_data["image_path"]
            source["image_base64"] = img_data["image_base64"]

        elif element_type == "table":
            source["table_markdown"] = _extract_table_markdown(content)

        sources.append(source)

    return sources


def generate(state: GraphState) -> GraphState:
    """
    Generate the final answer and populate cited_sources for the UI.
    """
    llm = ChatGroq(
        model       = GENERATION_MODEL,
        api_key     = GROQ_API_KEY,
        max_tokens  = 1024,
        temperature = 0.1,
    )

    docs    = state.get("graded_docs", [])
    context = _format_context_for_llm(docs)

    messages = []

    # System prompt + long-term memory
    system_content = _SYSTEM_PROMPT
    if state.get("memory_context"):
        system_content += f"\n\n{state['memory_context']}"
    messages.append(SystemMessage(content=system_content))

    # Last 3 chat turns for context
    for msg in state.get("chat_history", [])[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Current question + retry correction hint if applicable
    user_content = (
        f"Context Documents:\n{context}\n\n"
        f"Question: {state['query']}"
    )

    retry_count = state.get("retry_count", 0)
    if retry_count > 0:
        verdict = state.get("validation_result", "unknown")
        reason  = state.get("validation_reason", "no reason given")
        user_content += (
            f"\n\n⚠️  CORRECTION REQUIRED (Attempt {retry_count + 1}):\n"
            f"Your previous answer was rejected.\n"
            f"Problem type  : {verdict}\n"
            f"Specific issue: {reason}\n"
            f"Please correct your answer staying strictly within the context."
        )

    messages.append(HumanMessage(content=user_content))

    response = llm.invoke(messages)
    answer   = response.content.strip()

    # Build citation metadata for the UI
    cited_sources = _build_cited_sources(docs)

    logger.info(f"Generated answer ({len(answer)} chars), {len(cited_sources)} citations.")
    return {**state, "generation": answer, "cited_sources": cited_sources}