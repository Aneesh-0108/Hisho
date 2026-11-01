"""TransformComponent that restructures node metadata and prepares composite text for embedding."""

from typing import Sequence

from llama_index.core.schema import BaseNode, TransformComponent


def build_composite_searchable_text(
    letter_type: str | None,
    subject: str | None,
    description: str | None,
    body_content: str | None,
    max_body_chars: int = 1500,
) -> str:
    """Construct a composite searchable text representation for embedding.

    Combines letter type, subject, extracted description/purpose, and a focused
    slice of the authentic body content. Missing fields are omitted cleanly
    without throwing errors or injecting placeholder tokens.
    """
    sections: list[str] = []

    if letter_type and letter_type.strip():
        sections.append(f"Type: {letter_type.strip()}")

    if subject and subject.strip():
        sections.append(f"Subject: {subject.strip()}")

    if description and description.strip():
        sections.append(f"Purpose: {description.strip()}")

    if body_content and body_content.strip():
        cleaned_body = body_content.strip()
        if len(cleaned_body) > max_body_chars:
            cleaned_body = cleaned_body[:max_body_chars].rstrip()
        sections.append(f"Content:\n{cleaned_body}")

    if sections:
        return "\n\n".join(sections).strip()

    return (body_content or "").strip()


class NodeTextFormatter(TransformComponent):
    """Restructure node metadata and set node.text to composite text for vector search.

    Preserves:
    - ``node.metadata``: ref_no, date, subject, letter_type, primary_domain,
      description, table_info, approval_chain, signatures, full_letter,
      full_markdown, tab_title, source_doc_id, image_uris.
    - ``node.text``: Composite representation combining letter type, subject,
      purpose/description, and body content for Ollama nomic-embed-text embedding.
    """

    def __call__(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        for node in nodes:
            # 1. Extract and resolve all metadata fields safely
            ref_no = node.metadata.get("ref_no") or ""
            date = node.metadata.get("date") or ""
            subject = node.metadata.get("subject") or ""
            letter_type = node.metadata.get("letter_type") or ""
            primary_domain = node.metadata.get("primary_domain") or ""
            description = node.metadata.get("description") or ""
            table_info = node.metadata.get("table_info")
            approval_chain = node.metadata.get("approval_chain") or []
            signatures = node.metadata.get("signatures") or []
            tab_title = node.metadata.get("tab_title") or ""
            source_doc_id = node.metadata.get("source_doc_id") or ""
            image_uris = node.metadata.get("image_uris") or []

            # Full original letter preservation: never sacrifice authentic text
            full_letter = (
                node.metadata.get("full_markdown")
                or node.metadata.get("full_letter")
                or node.get_content()
                or ""
            )

            # 2. Update metadata with all required and preserved fields
            node.metadata = {
                "ref_no": ref_no,
                "date": date,
                "subject": subject,
                "letter_type": letter_type,
                "primary_domain": primary_domain,
                "description": description,
                "table_info": table_info,
                "approval_chain": approval_chain,
                "signatures": signatures,
                "full_letter": full_letter,
                "full_markdown": full_letter,
                "tab_title": tab_title,
                "source_doc_id": source_doc_id,
                "image_uris": image_uris,
            }

            # 3. Construct deterministic composite searchable representation
            node.text = build_composite_searchable_text(
                letter_type=letter_type,
                subject=subject,
                description=description,
                body_content=full_letter,
                max_body_chars=1500,
            )

        return nodes

    async def acall(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        return self.__call__(nodes, **kwargs)
