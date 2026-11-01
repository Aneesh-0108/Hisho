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

logger = logging.getLogger(__name__)


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
        """Sync nodes to legacy tables asynchronously."""
        if not nodes:
            logger.info("LegacyTableSync: no nodes to sync, skipping.")
            return list(nodes)

        # Check if there are any valid extracted nodes to sync
        has_valid_extraction = any(
            bool(n.metadata.get("letter_type") and n.metadata.get("description"))
            for n in nodes
        )
        if not has_valid_extraction:
            logger.warning(
                "LegacyTableSync: no nodes contain valid extraction metadata "
                "(letter_type and description). Skipping legacy sync to prevent "
                "destructive clearing of existing records."
            )
            return list(nodes)

        # Generate description-based embeddings for LetterTypeMemory
        description_embeddings = await self._generate_description_embeddings(nodes)

        # Collect source_doc_ids to scope signature deletions
        source_doc_ids = [
            n.metadata.get("source_doc_id")
            for n in nodes
            if n.metadata.get("source_doc_id")
        ]

        async with AsyncSessionLocal() as session:
            # Delete only signatures belonging to the documents currently being ingested
            if source_doc_ids:
                await session.execute(
                    delete(Signature).where(Signature.source_doc_id.in_(source_doc_ids))
                )
            else:
                await session.execute(delete(Signature))

            # Re-sync LetterTypeMemory with newly extracted records
            await session.execute(delete(LetterTypeMemory))

            existing_codes = await _load_existing_designations(session)

            added, designations_added = _sync_to_legacy_tables(
                session, nodes, existing_codes, description_embeddings
            )

            await session.commit()

        logger.info(
            "LegacyTableSync: created %d LetterTypeMemory records, "
            "%d new Designations.",
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

        memory = LetterTypeMemory(
            id=uuid.uuid4().hex,
            letter_type=meta.get("letter_type"),
            primary_domain=meta.get("primary_domain"),
            description=meta.get("description"),
            approval_chain=approval_chain,
            table_info=meta.get("table_info"),
            embedding=embedding,
            template_url=None,
        )
        session.add(memory)
        added += 1

        # 3. Signatures
        signatures = meta.get("signatures", [])
        source_doc_id = meta.get("source_doc_id", "")
        if signatures:
            for sig in signatures:
                if sig.get("url"):
                    session.add(
                        Signature(
                            id=uuid.uuid4().hex,
                            source_doc_id=source_doc_id,
                            image_url=sig["url"],
                            name=sig.get("name"),
                        )
                    )
                else:
                    logger.warning(
                        "LegacyTableSync: signature entry has no url: %s", sig
                    )

    return added, designations_added
