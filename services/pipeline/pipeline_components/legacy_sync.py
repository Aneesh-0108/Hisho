"""TransformComponent that syncs ingested nodes to legacy SQLAlchemy tables.

Handles three tables: ``LetterTypeMemory``, ``Designation``, and ``Signature``.

The ``LetterTypeMemory`` embedding is generated from the node's ``description``
metadata (not from ``node.text``), using a separate ``OllamaEmbedding`` call.
"""

from __future__ import annotations

import logging
import uuid
from typing import Sequence

from llama_index.core.schema import BaseNode, TransformComponent
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Designation, LetterTypeMemory, Signature
from services.pipeline.pipeline_components.embed_model import get_embed_model
from db.session import AsyncSessionLocal
from services.constants.letter_types import CANONICAL_CATEGORY_IDS

logger = logging.getLogger(__name__)


def _evaluate_signature_state(node: BaseNode) -> tuple[str, list[dict[str, Any]]]:
    """Evaluate signature extraction state for a node.

    Distinguishes three states:
      1. 'SUCCESS': Valid signature extraction with verified, non-empty signatures.
         -> Atomically replace existing source signatures with new ones.
      2. 'CONFIRMED_EMPTY': Confirmed document has zero signatures.
         -> Atomically delete existing source signatures, insert nothing.
      3. 'PRESERVE': Missing, failed, incomplete, or unverified extraction.
         -> Preserve existing source signatures without deletion or mutation.
    """
    meta = node.metadata
    status = meta.get("signature_extraction_status")
    signatures = meta.get("signatures")

    # State B: Explicitly confirmed zero signatures
    if status == "CONFIRMED_EMPTY":
        logger.info(
            "LegacyTableSync: Confirmed zero signatures for node %s. Existing signatures will be cleared.",
            node.node_id,
        )
        return "CONFIRMED_EMPTY", []

    # State A: Successful extraction with signatures
    if status == "SUCCESS":
        if isinstance(signatures, list) and len(signatures) > 0:
            valid_sigs: list[dict[str, Any]] = []
            for s in signatures:
                if isinstance(s, dict) and s.get("url") and isinstance(s["url"], str) and s["url"].strip():
                    valid_sigs.append(s)
                else:
                    logger.warning(
                        "LegacyTableSync: Incomplete signature record for node %s: %s. "
                        "Treating as incomplete; preserving existing signatures.",
                        node.node_id,
                        s,
                    )
                    return "PRESERVE", []
            return "SUCCESS", valid_sigs
        else:
            logger.warning(
                "LegacyTableSync: Node %s has status='SUCCESS' but signatures list is empty or invalid. "
                "Preserving existing signatures.",
                node.node_id,
            )
            return "PRESERVE", []

    # State C: Missing, failed, incomplete, or unverified extraction
    logger.info(
        "LegacyTableSync: Preserving existing signatures for node %s (status=%r, signatures=%s). "
        "Extraction is unverified or incomplete.",
        node.node_id,
        status,
        type(signatures).__name__ if signatures is not None else "None",
    )
    return "PRESERVE", []


class LegacyTableSync(TransformComponent):
    """Sync pipeline nodes to the legacy Signature, Designation, and LetterTypeMemory tables.

    This component runs inside the LlamaIndex ``IngestionPipeline`` as a
    ``TransformComponent``.  It opens its own async SQLAlchemy session,
    performs a full re-sync (delete-then-insert), and returns the nodes
    unchanged for downstream steps.

    The ``LetterTypeMemory.embedding`` column is populated with an embedding
    generated from the ``description`` metadata field, using the same Ollama
    embedding model configured in application settings.
    """

    def __call__(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        # Sync requires an async DB session; the pipeline must run via arun()/acall().
        raise NotImplementedError(
            "LegacyTableSync requires async execution. Use acall() or run "
            "the pipeline with arun()."
        )

    async def acall(
        self, nodes: Sequence[BaseNode], **kwargs
    ) -> Sequence[BaseNode]:
        """Sync nodes to legacy tables asynchronously with source scoping and validation."""
        if not nodes:
            logger.info("LegacyTableSync: no nodes to sync, skipping.")
            return list(nodes)

        # Validate extraction completeness for each node
        valid_indices: list[int] = []
        for idx, node in enumerate(nodes):
            meta = node.metadata or {}
            source_doc_id = meta.get("source_doc_id")
            if not source_doc_id and node.node_id and "#" in node.node_id:
                source_doc_id = node.node_id

            letter_type = meta.get("letter_type")
            description = meta.get("description")
            approval_chain = meta.get("approval_chain")

            if (
                source_doc_id
                and isinstance(source_doc_id, str)
                and source_doc_id.strip()
                and letter_type
                and letter_type in CANONICAL_CATEGORY_IDS
                and description
                and isinstance(description, str)
                and description.strip()
                and approval_chain is not None
                and isinstance(approval_chain, list)
            ):
                valid_indices.append(idx)

        if not valid_indices:
            logger.warning(
                "LegacyTableSync: no nodes contain valid, complete extraction metadata "
                "(source_doc_id, canonical letter_type, description, approval_chain). "
                "Skipping legacy sync to prevent destructive clearing of existing records."
            )
            return list(nodes)

        # Generate description-based embeddings for valid nodes
        valid_nodes = [nodes[i] for i in valid_indices]
        valid_embeddings = await self._generate_description_embeddings(valid_nodes)

        # Pair valid nodes with non-null embeddings
        ready_to_sync: list[tuple[BaseNode, list[float]]] = []
        for node, emb in zip(valid_nodes, valid_embeddings):
            if emb is not None:
                ready_to_sync.append((node, emb))

        if not ready_to_sync:
            logger.warning(
                "LegacyTableSync: embedding generation failed for all valid nodes. "
                "Skipping legacy sync."
            )
            return list(nodes)

        # Collect distinct, non-empty source_doc_ids to scope deletions
        scoped_source_ids: set[str] = set()
        for node, _ in ready_to_sync:
            sid = node.metadata.get("source_doc_id")
            if not sid and node.node_id and "#" in node.node_id:
                sid = node.node_id
            if sid and isinstance(sid, str) and sid.strip():
                scoped_source_ids.add(sid.strip())

        async with AsyncSessionLocal() as session:
            # Source-scoped cleanup: ONLY delete records belonging to successfully extracted sources.
            # Never issue a global DELETE FROM letter_type_memory or DELETE FROM signatures.
            # Legacy records with NULL source_doc_id are strictly preserved.
            for s_id in scoped_source_ids:
                await session.execute(
                    delete(LetterTypeMemory).where(LetterTypeMemory.source_doc_id == s_id)
                )

            existing_codes = await _load_existing_designations(session)

            nodes_to_sync = [n for n, _ in ready_to_sync]
            embs_to_sync = [emb for _, emb in ready_to_sync]

            added, designations_added = _sync_to_legacy_tables(
                session, nodes_to_sync, existing_codes, embs_to_sync
            )

            # Synchronize signatures based on explicit extraction state
            for node in nodes_to_sync:
                source_doc_id = node.metadata.get("source_doc_id") or node.node_id
                state, valid_sigs = _evaluate_signature_state(node)

                if state == "SUCCESS":
                    # State A: Replace existing signatures for this source atomically
                    await session.execute(
                        delete(Signature).where(Signature.source_doc_id == source_doc_id)
                    )
                    for sig in valid_sigs:
                        session.add(
                            Signature(
                                id=uuid.uuid4().hex,
                                source_doc_id=source_doc_id,
                                image_url=sig["url"],
                                name=sig.get("name"),
                            )
                        )
                elif state == "CONFIRMED_EMPTY":
                    # State B: Explicitly confirmed zero signatures — delete existing, insert nothing
                    await session.execute(
                        delete(Signature).where(Signature.source_doc_id == source_doc_id)
                    )
                elif state == "PRESERVE":
                    # State C: Preserve existing signatures — do NOT delete or mutate
                    pass

            await session.commit()

        logger.info(
            "LegacyTableSync: source-scoped sync completed for %d source(s) "
            "(%d LetterTypeMemory, %d new Designations).",
            len(scoped_source_ids),
            added,
            designations_added,
        )
        return list(nodes)

    @staticmethod
    async def _generate_description_embeddings(
        nodes: Sequence[BaseNode],
    ) -> list[list[float] | None]:
        """Generate embeddings from each node's description metadata.

        Returns a list aligned with ``nodes`` — one embedding per node, or
        ``None`` when description is empty / extraction failed.
        """
        embed_model = get_embed_model()

        descriptions: list[str] = []
        indices_with_desc: list[int] = []

        for i, node in enumerate(nodes):
            desc = node.metadata.get("description", "")
            if desc and node.metadata.get("letter_type"):
                descriptions.append(desc)
                indices_with_desc.append(i)

        # Batch embed all descriptions at once
        embeddings: list[list[float]] = []
        if descriptions:
            embeddings = await embed_model.aget_text_embedding_batch(descriptions)

        # Map back to per-node list
        result: list[list[float] | None] = [None] * len(nodes)
        for idx, emb in zip(indices_with_desc, embeddings):
            result[idx] = emb

        return result


async def _load_existing_designations(
    session: AsyncSession,
) -> dict[str, Designation]:
    """Return a title → Designation dict for dedup."""
    result = await session.execute(select(Designation))
    return {d.title: d for d in result.scalars()}


def _sync_to_legacy_tables(
    session: AsyncSession,
    nodes: Sequence[BaseNode],
    existing_codes: dict[str, Designation],
    description_embeddings: list[list[float] | None],
) -> tuple[int, int]:
    """Sync the ingested nodes to the legacy SQLAlchemy tables.

    Reads from the metadata populated by ``LetterMetadataExtractor`` and creates
    ``LetterTypeMemory``, ``Signature``, and ``Designation`` records.
    """
    added = 0
    designations_added = 0

    for i, node in enumerate(nodes):
        meta = node.metadata

        # Skip if metadata is empty (extraction failed)
        if not meta.get("letter_type"):
            continue

        # 1. Designations (preserve approval_chain dict without popping incumbent)
        approval_chain = meta.get("approval_chain", [])
        for step in approval_chain:
            incumbent = step.get("incumbent")
            desig_title = step.get("designation_title")
            if desig_title and desig_title not in existing_codes:
                desg = Designation(
                    id=uuid.uuid4().hex,
                    title=desig_title,
                    incumbent=incumbent,
                )
                session.add(desg)
                existing_codes[desig_title] = desg
                designations_added += 1

        # 2. LetterTypeMemory (embedding from description, not node.text)
        embedding = description_embeddings[i]
        if embedding is None:
            logger.warning(
                "LegacyTableSync: skipping LetterTypeMemory for node %s "
                "(no description embedding).",
                node.node_id,
            )
            continue

        source_doc_id = meta.get("source_doc_id") or node.node_id
        memory = LetterTypeMemory(
            id=uuid.uuid4().hex,
            source_doc_id=source_doc_id,
            letter_type=meta.get("letter_type"),
            primary_domain=meta.get("primary_domain") or "GENERAL",
            description=meta.get("description"),
            approval_chain=approval_chain,
            table_info=meta.get("table_info"),
            embedding=embedding,
            template_url=None,
        )
        session.add(memory)
        added += 1

    return added, designations_added
