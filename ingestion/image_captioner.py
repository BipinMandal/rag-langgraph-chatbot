"""
image_captioner.py — Generates text descriptions for images using a vision LLM.

Why do we need this?
Images (like Figure 1 — the Transformer architecture diagram) cannot be directly
searched as vectors. We convert each image to a text description so it becomes
semantically searchable.

Example:
  Before: content = "[IMAGE — description pending]"
  After:  content = "Caption: Figure 1: The Transformer model architecture.
                     Description: The diagram shows an encoder-decoder architecture.
                     The encoder (left) has 6 identical layers each containing
                     a multi-head self-attention mechanism and a feed-forward
                     network. The decoder (right) has an additional cross-attention
                     layer that attends to the encoder output..."
"""
from __future__ import annotations

import logging
from typing import List

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, VISION_MODEL
from ingestion.pdf_parser import DocumentElement

logger = logging.getLogger(__name__)

# Placeholder strings that indicate captioning has NOT been done yet.
# We use this set to filter out uncaptioned images before indexing.
PLACEHOLDER_CONTENT = {
    "[IMAGE — description pending]",
    "[IMAGE — no description available]",
}


def caption_images(elements: List[DocumentElement]) -> List[DocumentElement]:
    """
    Iterate through all DocumentElements.
    For each image that has base64 data, call the Groq vision model
    and replace the placeholder content with a rich text description.

    The LLM client is created once here and reused across all images.
    """
    # Only create the client if there are images to caption
    images_to_caption = [
        e for e in elements
        if e.element_type == "image" and e.image_base64
    ]

    if not images_to_caption:
        logger.info("No images to caption.")
        return elements

    logger.info(f"Captioning {len(images_to_caption)} images with Groq vision …")

    # Create the vision LLM client once (not inside the loop)
    llm = ChatGroq(model=VISION_MODEL, api_key=GROQ_API_KEY, max_tokens=512)

    for elem in images_to_caption:
        logger.info(f"  Captioning {elem.element_id} (page {elem.page_number}) …")

        # Build a context-aware prompt.
        # Including surrounding text helps the model give a better description.
        # For example, if the preceding text says "multi-head attention mechanism",
        # the model will focus on that aspect of the diagram.
        context_hint = ""
        if elem.preceding_context:
            context_hint += f"Text before this image: {elem.preceding_context[:400]}\n"
        if elem.caption:
            context_hint += f"Caption: {elem.caption}\n"
        if elem.section_header:
            context_hint += f"Section: {elem.section_header}\n"

        prompt = (
            f"{context_hint}\n"
            "Describe this image in detail for a technical audience:\n"
            "- What type of image is it? (diagram, chart, architecture, equation, etc.)\n"
            "- What are the key components, labels, arrows, or annotations?\n"
            "- What is the technical meaning or significance in this context?\n"
            "Keep the description under 300 words. Be specific and technical."
        )

        try:
            # Send image + prompt to the Groq vision model
            message = HumanMessage(content=[
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        # base64 data URL format required by the API
                        "url": f"data:image/jpeg;base64,{elem.image_base64}"
                    },
                },
            ])

            response    = llm.invoke([message])
            description = response.content.strip()

            # Build the final searchable content:
            # Caption + Description + Section context
            content_parts = []
            if elem.caption:
                content_parts.append(f"Caption: {elem.caption}")
            content_parts.append(f"Description: {description}")
            if elem.section_header:
                content_parts.append(f"(Section: {elem.section_header})")

            elem.content = "\n".join(content_parts)
            logger.info(f"    ✓ Done: {description[:80]} …")

        except Exception as error:
            # If captioning fails (e.g., API timeout), use caption text as fallback.
            # The placeholder remains if no caption is available either.
            logger.warning(f"    ✗ Captioning failed for {elem.element_id}: {error}")
            elem.content = elem.caption or "[IMAGE — no description available]"

    return elements