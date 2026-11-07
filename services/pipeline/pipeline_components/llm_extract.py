"""Extract structured letter metadata via a local Ollama model.

Provides:
- ``LetterMetadataExtractor``: A LlamaIndex ``BaseExtractor`` subclass for use
  inside an ``IngestionPipeline``.  Runs structured extraction via
  ``LLMTextCompletionProgram`` using the model specified by ``EXTRACTION_MODEL``,
  populates ``node.metadata`` with the extraction output, and **rewrites**
  ``node.text`` to contain only the generalized description so that downstream
  embeddings capture the concept — not document-specific names or dates.
- Pydantic schemas: ``LetterExtractionOutput``, ``ApprovalStep``, ``TableInfo``,
  ``SignatureInfo``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

from pydantic import BaseModel, Field, ValidationError

from llama_index.core.extractors import BaseExtractor
from llama_index.core.schema import BaseNode, TextNode
from llama_index.llms.ollama import Ollama
from llama_index.core.program import LLMTextCompletionProgram

from config import settings
from services.constants.letter_types import (
    format_letter_types_block,
    normalize_canonical_category,
)

logger = logging.getLogger(__name__)

_OLLAMA_MODEL = settings.extraction_model.removeprefix("ollama_chat/")

_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0


# ── Pydantic schemas ──────────────────────────────────────────────────────


class ApprovalStep(BaseModel):
    step_number: int = Field(description="Sequence number starting at 1")
    designation_title: str = Field(
        description="The designation name (e.g. 'Head of Department - CSE'). Omit college/company name. ONLY DESIGNATION NOT HIS PERSONAL NAME"
    )
    incumbent: str | None = Field(description="The name of the person holding the designation, if present.", default=None)


class TableInfo(BaseModel):
    summary: str | None = Field(description="High level summary, e.g. Budget breakdown", default=None)
    category: str | None = Field(description="<FINANCIAL | TIMELINE | ROSTER | INVENTORY | OTHER>", default=None)
    headers: list[str] = Field(
        description="List of column headers in the table, if any. Empty list if none.",
        default_factory=list
    )


class SignatureInfo(BaseModel):
    url: str | None = Field(description="URL of the image that acts as a signature", default=None)
    name: str | None = Field(description="The name of the signee.", default=None)


class LetterExtractionOutput(BaseModel):
    ref_no: str | None = Field(
        description="The reference number (ref no) of the letter, if any.", default=None
    )
    date: str | None = Field(
        description="The date of the letter, if any.", default=None
    )
    subject: str | None = Field(
        description="The subject line of the letter, if any.", default=None
    )
    letter_type: str = Field(
        description="Exactly one enum value from list above"
    )
    primary_domain: str = Field(
        description="<FINANCE | LOGISTICS | INFRASTRUCTURE | ACADEMICS | EVENT | GENERAL>"
    )
    description: str = Field(
        description=(
            "A clear, general-purpose summary using the full body that projects the letter's intent."
        )
    )
    table_info: TableInfo | None = Field(
        description="Information about the tabular data block, if one exists",
        default=None,
    )
    approval_chain: list[ApprovalStep] = Field(
        description=(
            "Ordered list of approvers from bottom (first) to top (final recipient)"
        )
    )
    signatures: list[SignatureInfo] = Field(
        description="List of identified signatures and their designations",
        default_factory=list
    )


# ── System prompt ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
You are a structured-data extraction assistant for an institutional document archiving system. 
Analyze the provided Markdown document (which contains text, tables, and image tokens) and output ONLY a JSON object conforming strictly to the schema below.

Categories for `letter_type` (Choose EXACTLY ONE based on Intent + Domain):
{letter_types}

General Schema & Extraction Rules:
- DO NOT default to a generic 'REQUEST' for letter_type. Always qualify it using the subject and target domain.
- `approval_chain` MUST be bottom-up (Step 1 is the first endorser, final step is the top recipient).
- `signatures` MUST ONLY include URLs of images that act as signatures (typically found at the end of the document or near approval titles), ignoring logos, letterheads, or structural images. It MUST also include the name of the signee.
- Image URLs appear as [SIGNATURE_IMAGE: SIG_N] tokens. Map them back to the image_uris in the metadata if available.

CRITICAL RULES FOR `description`:
1. The `description` field will be directly converted into a vector for concept matching and document intent search.
2. It MUST NOT contain specific names, dates (e.g., 13 July 2026), amounts, room numbers, specific software, hardware models, or specific event titles.
3. You MUST construct `description` using this exact sentence structure formula:
   "Formal request seeking [ACTION PHRASE] [GENERIC ENTITY/CATEGORY] for [GENERIC PURPOSE/INTENT]."

Abstraction & Generalized Vocabulary Rules:
- Specific Hardware/Software (e.g., "NVIDIA Jetson Nano", "Claude SDK", "Hostinger", "1TB NVMe") -> Replace with "hardware developer kits", "software API licenses", or "cloud infrastructure".
- Specific Venues/Rooms (e.g., "Prerna Hall", "Room 3410", "CCC") -> Replace with "an institutional hall", "a department classroom", or "a campus facility".
- Specific Events/Activities (e.g., "Winter Hackathon", "IndiaFOSS", "The Vanguard ARG", "Code Sprint") -> Replace with "a technical competition", "a community exhibition", "an interactive student activity", or "an academic workshop".
- Specific Equipment/Materials (e.g., "Camera, Tripod, Cables", "Sound System", "Water bottles, Staplers") -> Replace with "audiovisual and streaming equipment", "event sound logistics", or "event operational stationery".
- Specific Infrastructure/Repairs (e.g., "broken tiles, door, rat infestation, wiring") -> Replace with "physical clubroom infrastructure and workspace facilities".

FEW-SHOT EXAMPLES FOR `description`:
---
Input: "Requesting release of Rs 66,000 for NVIDIA Jetson Nano ML kit and NVMe SSD for AI projects."
Description: "Formal request seeking sanction and release of funds for the procurement of specialized hardware infrastructure and developer kits for technical training."
---
Input: "Seeking permission to use Room 3410 and borrow camera setup to stream the OpenStreetMap event."
Description: "Formal request seeking venue allocation and temporary borrowing of audiovisual equipment to broadcast an academic workshop."
---
Input: "Requesting reimbursement of Rs 22,000 spent for Claude 1 Year subscription from approved budget."
Description: "Formal request seeking reimbursement of prior-approved institutional funds incurred for software subscription licenses and digital development services."
---
Input: "Requesting complete renovation of the Sceptix club room due to broken tiles, doors, wiring, and rat infestation."
Description: "Formal request seeking administrative authorization for comprehensive structural, electrical, and facility renovation of a student workspace."
---

Output formatting:
- Output raw JSON only. No markdown formatting, no backticks, no commentary.

Document to analyze:
{letter_markdown}

"""

_SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.format(
    letter_types=format_letter_types_block(),
    letter_markdown="{letter_markdown}",
)


# ── LlamaIndex LLM + structured extraction program ──────────────────────

_llm = Ollama(
    model=_OLLAMA_MODEL,
    base_url=settings.ollama_base_url,
    request_timeout=120.0,
    temperature=0.0,
    thinking=False,
    json_mode=True,
)

_extraction_program = LLMTextCompletionProgram.from_defaults(
    output_cls=LetterExtractionOutput,
    prompt_template_str=_SYSTEM_PROMPT,
    llm=_llm,
    verbose=False,
)


# ── BaseExtractor subclass ───────────────────────────────────────────────


class LetterMetadataExtractor(BaseExtractor):
    """LlamaIndex ``BaseExtractor`` that runs Ollama structured extraction.

    For each node:
    1. Sends the node's Markdown text to ``LLMTextCompletionProgram``.
    2. Validates the response into ``LetterExtractionOutput``.
    3. Returns a metadata dict with all extracted fields.

    After extraction, ``aprocess_nodes`` rewrites ``node.text`` to contain
    only the generalized ``description`` so that downstream embedding models
    vectorise the concept, not document-specific names or dates.
    """

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict[str, Any]]:
        """Extract structured metadata from each node via LLM."""
        metadata_list: list[dict[str, Any]] = []

        for node in nodes:
            content = node.get_content()
            if not content.strip():
                metadata_list.append({})
                continue

            extraction = await self._extract_with_retry(content)
            if extraction is None:
                # Extraction failed after all retries — pass through empty metadata
                logger.error(
                    "LetterMetadataExtractor: extraction failed for node %s after %d retries",
                    node.node_id,
                    _MAX_RETRIES,
                )
                metadata_list.append({"signature_extraction_status": "FAILED"})
                continue

            # Map signature tokens back to actual image URIs from node metadata
            image_uris: list[str] = node.metadata.get("image_uris", [])
            resolved_signatures = self._resolve_signature_uris(
                extraction.signatures, image_uris
            )

            # Determine explicit signature extraction status
            existing_sig_status = node.metadata.get("signature_extraction_status")
            if existing_sig_status:
                sig_status = existing_sig_status
            elif resolved_signatures and any(s.url for s in resolved_signatures):
                sig_status = "SUCCESS"
            else:
                # Unverified / conservative: do not treat empty extraction as confirmed absence
                sig_status = "UNVERIFIED"

            # Normalize letter_type against canonical taxonomy
            doc_title = node.metadata.get("tab_title") or node.metadata.get("title") or ""
            try:
                canonical_type = normalize_canonical_category(
                    extraction.letter_type, document_title=doc_title
                )
            except ValueError as exc:
                logger.error(
                    "LetterMetadataExtractor: category normalization failed for node %s: %s",
                    node.node_id,
                    exc,
                )
                metadata_list.append({"signature_extraction_status": "FAILED"})
                continue

            metadata_list.append(
                {
                    "ref_no": extraction.ref_no,
                    "date": extraction.date,
                    "subject": extraction.subject,
                    "letter_type": canonical_type,
                    "primary_domain": extraction.primary_domain,
                    "description": extraction.description,
                    "table_info": extraction.table_info.model_dump() if extraction.table_info else None,
                    "approval_chain": [
                        step.model_dump() for step in extraction.approval_chain
                    ],
                    "signatures": [
                        sig.model_dump() for sig in resolved_signatures
                    ],
                    "signature_extraction_status": sig_status,
                }
            )

        return metadata_list

    async def aprocess_nodes(
        self,
        nodes: list[BaseNode],
        excluded_embed_metadata_keys: list[str] | None = None,
        excluded_llm_metadata_keys: list[str] | None = None,
        **kwargs: Any,
    ) -> list[BaseNode]:
        """Run base extraction, then rewrite ``node.text`` to the description only."""
        nodes = await super().aprocess_nodes(
            nodes,
            excluded_embed_metadata_keys=excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=excluded_llm_metadata_keys,
            **kwargs,
        )

        for node in nodes:
            description = node.metadata.get("description", "")
            if description:
                # Rewrite node text to the generalized summary only.
                # This guarantees PGVector embeds the *concept*, not raw content.
                node.text = description

        return nodes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _extract_with_retry(
        content: str,
    ) -> LetterExtractionOutput | None:
        """Call the LLM extraction program with exponential-backoff retries.

        Handles ``ValidationError`` (malformed JSON / missing fields) and
        generic exceptions to prevent single-document failures from crashing
        the entire batch.
        """
        backoff = _INITIAL_BACKOFF
        last_error: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result: LetterExtractionOutput = await asyncio.to_thread(
                    _extraction_program, letter_markdown=content[:12000]
                )
                return result

            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    "LetterMetadataExtractor: Pydantic validation failed (attempt %d/%d): %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LetterMetadataExtractor: unexpected error (attempt %d/%d): %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )

            if attempt < _MAX_RETRIES:
                await asyncio.sleep(backoff)
                backoff *= 2

        logger.error(
            "LetterMetadataExtractor: all %d retries exhausted. Last error: %s",
            _MAX_RETRIES,
            last_error,
        )
        return None

    @staticmethod
    def _resolve_signature_uris(
        signatures: list[SignatureInfo],
        image_uris: list[str],
    ) -> list[SignatureInfo]:
        """Map signature image token references back to actual URIs.

        The GoogleDocJsonParser replaces images with ``[SIGNATURE_IMAGE: SIG_N]``
        tokens.  The LLM may reference these in its output.  This method attempts
        to pair each signature entry with the corresponding real URI from the
        document's ``image_uris`` metadata.
        """
        resolved: list[SignatureInfo] = []
        # Build a quick index of signature-position URIs (those tagged SIG_n)
        # In the parser, signature images are the *last* URIs added to image_uris
        # when _is_likely_signature returns True.  We use a simple heuristic:
        # assign URIs from the end of the list to the signatures.
        sig_uris = list(reversed(image_uris))  # most recent = most likely sig

        for i, sig in enumerate(signatures):
            url = sig.url
            # If the LLM returned a SIG_N token instead of a real URL, resolve it
            if url and url.startswith("[SIGNATURE_IMAGE"):
                url = sig_uris[i] if i < len(sig_uris) else None
            elif url and not url.startswith("http"):
                # LLM returned something we can't use — try positional fallback
                url = sig_uris[i] if i < len(sig_uris) else None

            resolved.append(SignatureInfo(url=url, name=sig.name))

        return resolved
