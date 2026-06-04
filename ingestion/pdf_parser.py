"""
pdf_parser.py — Extracts and structures all content from a PDF file.

What this module does (step by step):
  1. Uses 'unstructured' library to detect element types (title, text, table, image)
  2. Uses PyMuPDF to extract raw image bytes (for vision captioning later)
  3. Tracks document hierarchy (which section each element belongs to)
  4. Attaches surrounding context to every image and table
     so they are NEVER stored in isolation (orphaned)
  5. Detects equations and attaches them to surrounding context
  6. Finds cross-references like "see Figure 1" or "refer to Section 3.2"

FIXES APPLIED:
  ✅ Hierarchy heuristic removed — no longer uses text length to guess depth.
     Now uses category_depth from unstructured (reliable), defaults to 1 if absent.
  ✅ Equation detection added — math content is treated like images:
     stored with surrounding context, never as isolated noise.
  ✅ BeautifulSoup import moved to top of file (was inside a loop body).
  ✅ Image cursor mapping is now more robust — falls back gracefully if
     PyMuPDF finds fewer images than unstructured on a given page.
"""
from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF — for extracting image bytes
from bs4 import BeautifulSoup  # for parsing HTML tables
from unstructured.documents.elements import Image as UImage
from unstructured.documents.elements import (
    ListItem,
    NarrativeText,
    Table,
    Text,
    Title,
)
from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# How many paragraphs before/after an image or table to capture as context
_CONTEXT_WINDOW = 2

# Short text immediately adjacent to an image/table is likely a caption
_CAPTION_MAX_LEN  = 220
_CAPTION_KEYWORDS = ("figure", "fig.", "table", "tab.", "equation", "eq.")

# Regex to find cross-references in text, e.g.:
#   "see Figure 1",  "as shown in Table 3",  "refer to Section 2.1"
_XREF_PATTERN = re.compile(
    r"\b(see|refer to|as shown in|as described in|in|from)\s+"
    r"(figure|fig\.|table|tab\.|section|eq\.|equation)\s*[\d\.]+",
    re.IGNORECASE,
)

# Simple heuristic for detecting equations:
# LaTeX-style commands or lines with many special math characters
_EQUATION_PATTERN = re.compile(
    r"(\\frac|\\sum|\\int|\\alpha|\\beta|\\theta|\\sigma|\\mu"
    r"|\$.*?\$"                      # inline LaTeX: $...$
    r"|^\s*\([0-9]+\)\s*$"           # numbered equations: (1), (2)
    r"|softmax\s*\(|Attention\s*\()" # paper-specific equation forms
)


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class DocumentElement:
    """
    A single structured piece of content extracted from the PDF.

    Think of this as one "card" of information:
    - A paragraph of text
    - A table (converted to Markdown)
    - An image (will have a description added later)
    - An equation (with its surrounding explanation)
    """
    element_id:        str
    element_type:      str          # 'text' | 'title' | 'table' | 'image' | 'equation'
    content:           str          # the actual searchable text content
    raw_content:       str  = ""    # original table markdown (before compositing)
    page_number:       int  = 0
    section_header:    str  = ""    # nearest section title above this element
    hierarchy:         List[str] = field(default_factory=list)  # [H1, H2, H3, ...]
    preceding_context: str  = ""    # paragraph(s) before this element
    following_context: str  = ""    # paragraph(s) after this element
    caption:           str  = ""    # inline caption (e.g., "Figure 1: ...")
    image_base64:      Optional[str] = None   # JPEG/PNG bytes as base64 string
    image_path:        Optional[str] = None   # path to saved image file
    metadata:          Dict = field(default_factory=dict)


# ── Main parser class ─────────────────────────────────────────────────────────

class PDFParser:
    """
    Parses a PDF into a list of DocumentElement objects.

    Usage:
        parser   = PDFParser("paper.pdf")
        elements = parser.parse()
        # elements is a list of DocumentElement objects ready for chunking
    """

    def __init__(self, pdf_path: str, image_dir: str = "extracted_images"):
        self.pdf_path  = Path(pdf_path)
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        # We track the current section hierarchy as we scan through elements.
        # Example: ["Introduction", "Background", "Related Work"]
        self._hierarchy: List[str] = []

    # ── Public entry point ────────────────────────────────────────────────

    def parse(self) -> List[DocumentElement]:
        """
        Full parsing pipeline. Returns all elements ready for chunking.
        """
        # Step 1: Use unstructured to detect all elements in the PDF
        logger.info("Step 1/5 — Partitioning PDF with unstructured (hi_res mode) …")
        logger.info("  (This may take 1-3 minutes for a 15-page paper — it's doing OCR)")
        raw_elements = partition_pdf(
            filename                       = str(self.pdf_path),
            strategy                       = "hi_res",       # best quality
            extract_images_in_pdf          = True,           # save images to image_dir
            extract_image_block_output_dir = str(self.image_dir),
            infer_table_structure          = True,           # get table HTML
            include_page_breaks            = False,
        )
        logger.info(f"  Found {len(raw_elements)} raw elements.")

        # Step 2: Extract image bytes with PyMuPDF for vision captioning
        logger.info("Step 2/5 — Extracting image bytes with PyMuPDF …")
        page_images = self._extract_images_pymupdf()

        # Step 3: Convert raw elements into structured DocumentElement objects
        logger.info("Step 3/5 — Building structured DocumentElement list …")
        elements = self._build_elements(raw_elements, page_images)

        # Step 4: Attach surrounding context to images, tables, equations
        logger.info("Step 4/5 — Attaching context to images/tables/equations …")
        elements = self._attach_context(elements)

        # Step 5: Find cross-references like "see Figure 1"
        logger.info("Step 5/5 — Extracting cross-references …")
        elements = self._extract_cross_references(elements)

        logger.info(f"✓ Parsing complete — {len(elements)} structured elements.")
        return elements

    # ── Step 2: PyMuPDF image extraction ─────────────────────────────────

    def _extract_images_pymupdf(self) -> Dict[int, List[Dict]]:
        """
        Use PyMuPDF to extract raw image bytes from every page.

        Returns a dict: {page_number: [{"base64": ..., "path": ..., "ext": ...}]}

        Why do we use PyMuPDF here AND unstructured above?
        - unstructured detects WHERE images are in the layout
        - PyMuPDF gives us the actual image bytes (base64) needed for the vision model
        """
        result: Dict[int, List[Dict]] = {}
        doc = fitz.open(str(self.pdf_path))

        for page_no, page in enumerate(doc, start=1):
            page_imgs = []
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref      = img_info[0]                    # internal image reference
                base_img  = doc.extract_image(xref)        # extract raw bytes
                img_bytes = base_img["image"]
                ext       = base_img["ext"]                # "jpeg", "png", etc.

                # Save image to disk (useful for debugging)
                out_path = self.image_dir / f"page{page_no}_img{img_idx}.{ext}"
                out_path.write_bytes(img_bytes)

                page_imgs.append({
                    "base64": base64.b64encode(img_bytes).decode("utf-8"),
                    "path":   str(out_path),
                    "ext":    ext,
                })

            if page_imgs:
                result[page_no] = page_imgs

        doc.close()
        logger.info(f"  Extracted images from {len(result)} pages.")
        return result

    # ── Step 3: Build element list ────────────────────────────────────────

    @staticmethod
    def _classify_type(raw) -> str:
        """
        Map an unstructured element to our simplified type string.

        unstructured has many element types (Title, NarrativeText, ListItem,
        Formula, Image, Table, etc.). We simplify to 5 types.
        """
        if isinstance(raw, Title):                           return "title"
        if isinstance(raw, Table):                           return "table"
        if isinstance(raw, UImage):                          return "image"
        if isinstance(raw, (NarrativeText, Text, ListItem)): return "text"
        return "text"   # default: treat unknown types as plain text

    def _update_hierarchy(self, raw_element, element_type: str) -> List[str]:
        """
        Maintain a stack representing the current heading path.

        FIX APPLIED: No longer uses text length as a depth heuristic.
        Uses category_depth from unstructured (reliable for hi_res mode).
        Defaults to depth=1 (subheading) if category_depth is not available.

        Example result: ["3 Architecture", "3.2 Attention", "3.2.1 Scaled Dot-Product"]
        """
        if element_type != "title":
            # Non-title elements don't change the hierarchy — just return current
            return list(self._hierarchy)

        # Get the heading depth from unstructured metadata
        # depth=0 means top-level (H1), depth=1 means H2, etc.
        depth = getattr(raw_element.metadata, "category_depth", None)
        if depth is None:
            # FIX: default to 1 (subheading) instead of guessing from length
            # This is safer — misidentifying a subheading as H1 creates false hierarchy
            depth = 1

        # Extend or truncate the hierarchy stack to this depth
        # Example: if current hierarchy is ["H1", "H2"] and depth=1 (H2),
        #          we keep ["H1"] and set position 1 to new heading
        while len(self._hierarchy) <= depth:
            self._hierarchy.append("")
        self._hierarchy = self._hierarchy[:depth + 1]
        self._hierarchy[depth] = raw_element.text.strip()

        return list(self._hierarchy)

    @staticmethod
    def _is_equation(text: str) -> bool:
        """
        Detect if a text element is primarily a mathematical equation.

        Why handle equations separately?
        The "Attention Is All You Need" paper has equations like:
          Attention(Q, K, V) = softmax(QK^T / √dk) V
        These are meaningless as standalone text chunks but become searchable
        when stored with the surrounding explanation.
        """
        return bool(_EQUATION_PATTERN.search(text))

    @staticmethod
    def _table_to_markdown(raw_table) -> str:
        """
        Convert an unstructured Table element to a Markdown table string.

        Why Markdown?
        - Human-readable format that LLMs understand well
        - Preserves row/column structure
        - Can be embedded as text in the vector store

        Example output:
            | Model | BLEU | Params |
            |-------|------|--------|
            | Base  | 27.3 | 65M    |
        """
        html = getattr(raw_table.metadata, "text_as_html", None)
        if not html:
            return raw_table.text   # fallback to plain text if no HTML

        soup = BeautifulSoup(html, "html.parser")
        rows = soup.find_all("tr")
        if not rows:
            return raw_table.text

        md_lines: List[str] = []
        for row_idx, row in enumerate(rows):
            cells     = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
            md_lines.append("| " + " | ".join(cells) + " |")
            # Add separator row after the header
            if row_idx == 0:
                md_lines.append("|" + "|".join(["---"] * len(cells)) + "|")

        return "\n".join(md_lines)

    def _build_elements(
        self,
        raw_elements: List,
        page_images:  Dict[int, List[Dict]],
    ) -> List[DocumentElement]:
        """
        Convert unstructured's raw element list into our DocumentElement list.

        This is the core loop that processes every detected element:
        - text/title → stored as-is
        - table       → converted to Markdown
        - image       → linked to PyMuPDF image bytes
        - equation    → detected in text, typed separately
        """
        elements: List[DocumentElement] = []

        # Track which image index we've consumed per page
        # (images are extracted in page order, so we match them sequentially)
        img_cursor: Dict[int, int] = {}

        for idx, raw in enumerate(raw_elements):
            element_type = self._classify_type(raw)
            page_no      = int(getattr(raw.metadata, "page_number", 0) or 0)
            hierarchy    = self._update_hierarchy(raw, element_type)

            # The "nearest ancestor header" = the last non-empty item in hierarchy
            section = next((h for h in reversed(hierarchy) if h), "")

            # ── Text / Title elements ──────────────────────────────────────
            if element_type in ("text", "title"):
                text = raw.text.strip()
                if not text:
                    continue   # skip empty elements

                # Check if this text is actually an equation
                # We reclassify it so it gets anchor context (not split)
                if element_type == "text" and self._is_equation(text):
                    elements.append(DocumentElement(
                        element_id     = f"equation_{idx}",
                        element_type   = "equation",
                        content        = text,
                        page_number    = page_no,
                        section_header = section,
                        hierarchy      = hierarchy,
                    ))
                else:
                    elements.append(DocumentElement(
                        element_id     = f"{element_type}_{idx}",
                        element_type   = element_type,
                        content        = text,
                        page_number    = page_no,
                        section_header = section,
                        hierarchy      = hierarchy,
                    ))

            # ── Table elements ─────────────────────────────────────────────
            elif element_type == "table":
                md_table = self._table_to_markdown(raw)
                elements.append(DocumentElement(
                    element_id     = f"table_{idx}",
                    element_type   = "table",
                    content        = f"[TABLE]\n{md_table}",
                    raw_content    = md_table,
                    page_number    = page_no,
                    section_header = section,
                    hierarchy      = hierarchy,
                    metadata       = {"html": getattr(raw.metadata, "text_as_html", "")},
                ))

            # ── Image elements ─────────────────────────────────────────────
            elif element_type == "image":
                # Match this unstructured image to the PyMuPDF-extracted bytes.
                # Both extract images in page order, so we use a cursor per page.
                cursor    = img_cursor.get(page_no, 0)
                page_imgs = page_images.get(page_no, [])

                img_b64  = None
                img_path = None
                if cursor < len(page_imgs):
                    img_b64  = page_imgs[cursor]["base64"]
                    img_path = page_imgs[cursor]["path"]
                    img_cursor[page_no] = cursor + 1
                else:
                    # FIX: Graceful fallback — don't crash if counts differ.
                    # This can happen with SVGs or vector graphics that
                    # PyMuPDF counts differently than unstructured.
                    logger.warning(
                        f"  Image {idx} on page {page_no}: no matching PyMuPDF "
                        f"image found (cursor={cursor}, available={len(page_imgs)}). "
                        f"Will store without base64 — captioning will be skipped."
                    )

                elements.append(DocumentElement(
                    element_id     = f"image_{idx}",
                    element_type   = "image",
                    content        = "[IMAGE — description pending]",  # filled by captioner
                    page_number    = page_no,
                    section_header = section,
                    hierarchy      = hierarchy,
                    image_base64   = img_b64,
                    image_path     = img_path,
                ))

        return elements

    # ── Step 4: Attach surrounding context ───────────────────────────────

    def _attach_context(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """
        For every image, table, and equation — attach the surrounding paragraphs.

        WHY this matters:
        If the paper says "as shown in Figure 1, the encoder uses multi-head attention"
        and then Figure 1 appears, we want to store:
            [The encoder uses multi-head attention (preceding)]
            [Figure 1: Architecture diagram (content)]
            [The decoder is connected via cross-attention (following)]

        This way, when someone asks "how does the encoder work?", the vector
        search finds the composite chunk — not just an isolated image.

        Captions are detected and stored separately from context.
        A caption is a SHORT piece of text immediately adjacent to the image/table
        that contains words like "figure", "table", "fig.", etc.
        """
        n = len(elements)

        for i, elem in enumerate(elements):
            if elem.element_type not in ("image", "table", "equation"):
                continue

            # ── Capture preceding context ──────────────────────────────────
            pre_parts: List[str] = []
            caption_before = ""
            j, text_count  = i - 1, 0

            while j >= 0 and text_count < _CONTEXT_WINDOW:
                prev = elements[j]
                if prev.element_type in ("text", "title"):
                    txt = prev.content.strip()

                    # Is this immediately-preceding text a caption?
                    is_caption = (
                        j == i - 1
                        and len(txt) <= _CAPTION_MAX_LEN
                        and any(kw in txt.lower() for kw in _CAPTION_KEYWORDS)
                    )
                    if is_caption:
                        caption_before = txt
                    else:
                        pre_parts.insert(0, txt)   # insert at front to preserve order
                        text_count += 1
                j -= 1

            # ── Capture following context ──────────────────────────────────
            fol_parts: List[str] = []
            caption_after = ""
            j, text_count = i + 1, 0

            while j < n and text_count < _CONTEXT_WINDOW:
                nxt = elements[j]
                if nxt.element_type in ("text", "title"):
                    txt = nxt.content.strip()

                    is_caption = (
                        j == i + 1
                        and len(txt) <= _CAPTION_MAX_LEN
                        and any(kw in txt.lower() for kw in _CAPTION_KEYWORDS)
                    )
                    if is_caption:
                        caption_after = txt
                    else:
                        fol_parts.append(txt)
                        text_count += 1
                j += 1

            elem.preceding_context = " ".join(pre_parts)
            elem.following_context = " ".join(fol_parts)
            elem.caption           = caption_before or caption_after

        return elements

    # ── Step 5: Cross-reference extraction ───────────────────────────────

    def _extract_cross_references(
        self, elements: List[DocumentElement]
    ) -> List[DocumentElement]:
        """
        Find all phrases like "see Figure 1", "refer to Section 3.2",
        "as shown in Table 2" and store them in each element's metadata.

        Why store cross-references?
        When a user asks "explain Table 2", we want to boost retrieval
        of chunks that mention Table 2 — even if those chunks don't
        contain Table 2 itself.

        The grader uses this metadata to slightly boost scores for
        documents that reference what the user is asking about.
        """
        for elem in elements:
            matches = _XREF_PATTERN.findall(elem.content)
            if matches:
                # Normalise to lowercase for easy matching later
                # e.g., [("see", "figure 1"), ...] → ["see figure 1", ...]
                xrefs = [" ".join(m).lower().strip() for m in matches]
                elem.metadata["cross_refs"] = xrefs
                logger.debug(f"  Cross-refs in {elem.element_id}: {xrefs}")

        return elements