"""Custom LlamaIndex NodeParser: Google Doc JSON → clean Markdown TextNodes.

Transforms the raw JSON payload produced by ``DocumentTransformer.transform_to_payload()``
(a list of ``{type: text|table|image, ...}`` dicts) into a token-efficient Markdown
string, while preserving raw image URIs and document metadata in ``node.metadata``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import BaseNode, TextNode

logger = logging.getLogger(__name__)


class GoogleDocJsonParser(NodeParser):
    """Parse Google Doc JSON payloads into clean Markdown TextNodes.

    Expected input: ``Document`` objects where ``doc.text`` is the JSON string
    returned by ``DocumentTransformer.transform_to_payload()`` — a list of
    element dicts with ``type`` in ``{"text", "table", "image"}``.

    Output: one ``TextNode`` per document with:
      - ``node.text`` = clean Markdown representation
      - ``node.metadata`` gains ``image_uris`` (list[str]) and
        ``full_markdown`` (str, copy of the converted text)
    """

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[BaseNode]:
        parsed: list[BaseNode] = []

        for node in nodes:
            raw_text = node.get_content()

            # --- Deserialise the JSON payload ---
            try:
                elements: list[dict] = json.loads(raw_text)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "GoogleDocJsonParser: non-JSON content in node %s — passing through as-is",
                    node.node_id,
                )
                parsed.append(node)
                continue

            if not isinstance(elements, list):
                logger.warning(
                    "GoogleDocJsonParser: expected list, got %s — skipping node %s",
                    type(elements).__name__,
                    node.node_id,
                )
                continue

            # --- Convert to Markdown ---
            markdown_parts: list[str] = []
            image_uris: list[str] = []
            sig_counter = 0
            img_counter = 0

            for idx, el in enumerate(elements):
                el_type = el.get("type", "")

                if el_type == "text":
                    content = el.get("content", "").strip()
                    if content:
                        markdown_parts.append(content)

                elif el_type == "table":
                    table_md = self._table_to_markdown(el.get("rows", []))
                    if table_md:
                        markdown_parts.append(table_md)

                elif el_type == "image":
                    uri = el.get("url", "")
                    if uri:
                        image_uris.append(uri)
                        if self._is_likely_signature(idx, elements):
                            sig_counter += 1
                            markdown_parts.append(
                                f"[SIGNATURE_IMAGE: SIG_{sig_counter}]"
                            )
                        else:
                            img_counter += 1
                            markdown_parts.append(
                                f"[IMAGE: IMG_{img_counter}]"
                            )

            markdown_text = "\n\n".join(markdown_parts).strip()

            if not markdown_text:
                continue

            # --- Build the output TextNode ---
            new_metadata = dict(node.metadata)  # inherit parent metadata
            new_metadata["image_uris"] = image_uris
            new_metadata["full_markdown"] = markdown_text

            stable_id = new_metadata.get("source_doc_id") or node.node_id
            text_node = TextNode(
                id_=stable_id,
                text=markdown_text,
                metadata=new_metadata,
                excluded_embed_metadata_keys=[
                    "image_uris",
                    "full_markdown",
                    "source_doc_id",
                    "tab_title",
                ],
                excluded_llm_metadata_keys=[
                    "image_uris",
                    "full_markdown",
                ],
            )
            # Preserve lineage to the source document
            text_node.relationships = node.relationships

            parsed.append(text_node)

        return parsed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _table_to_markdown(rows: list[list[str]]) -> str:
        """Convert a list-of-lists table into a standard Markdown grid table."""
        if not rows:
            return ""

        # Normalise column count to the widest row
        col_count = max(len(r) for r in rows)
        normalised = [r + [""] * (col_count - len(r)) for r in rows]

        lines: list[str] = []
        # Header row
        header = normalised[0]
        lines.append("| " + " | ".join(cell.strip() for cell in header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")

        # Data rows
        for row in normalised[1:]:
            lines.append("| " + " | ".join(cell.strip() for cell in row) + " |")

        return "\n".join(lines)

    @staticmethod
    def _is_likely_signature(image_idx: int, elements: list[dict]) -> bool:
        """Heuristic: an image is likely a signature if it appears in the bottom
        half of the document, or anywhere in a document containing
        signature-indicator keywords."""
        total = len(elements)
        # Located in the bottom half of the document/payload
        if image_idx >= total * 0.5:
            return True

        # Signature keywords/closing phrases
        sig_keywords = {
            "signature",
            "signed",
            "sign",
            "yours",
            "sincerely",
            "faithfully",
            "regards",
            "thanking",
            "sd/-",
            "secretary",
            "president",
            "vice-president",
            "coordinator",
            "director",
            "principal",
        }

        # Check if document contains any of the keywords
        # (This implicitly covers "in close proximity" as well)
        for el in elements:
            if el.get("type") == "text":
                text_lower = el.get("content", "").lower()
                if any(kw in text_lower for kw in sig_keywords):
                    return True

        return False
