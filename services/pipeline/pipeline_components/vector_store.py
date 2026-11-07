"""PGVectorStore construction with idempotent upsert support."""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any, List

from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.postgres import PGVectorStore
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config import settings

logger = logging.getLogger(__name__)

VECTOR_TABLE_NAME = "letter_nodes"


class IdempotentPGVectorStore(PGVectorStore):
    """Subclass of PGVectorStore that performs atomic PostgreSQL upserts.

    PGVectorStore 0.8.1 defaults to plain INSERT via session.add(), which fails
    on re-ingesting documents with a unique constraint on node_id.
    This class overrides async_add and add to perform atomic ON CONFLICT DO UPDATE
    operations on node_id, guaranteeing safe, idempotent re-ingestion without data loss.
    """

    async def async_add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        """Atomically upsert nodes by node_id asynchronously."""
        self._initialize()
        ids: list[str] = []
        async with self._async_session() as session:
            async with session.begin():
                for node in nodes:
                    ids.append(node.node_id)
                    metadata_dict = node_to_metadata_dict(
                        node,
                        remove_text=True,
                        flat_metadata=self.flat_metadata,
                    )
                    stmt = pg_insert(self._table_class).values(
                        node_id=node.node_id,
                        embedding=node.get_embedding(),
                        text=node.get_content(metadata_mode=MetadataMode.NONE),
                        metadata_=metadata_dict,
                    )
                    upsert_stmt = stmt.on_conflict_do_update(
                        index_elements=["node_id"],
                        set_={
                            "text": stmt.excluded.text,
                            "metadata_": stmt.excluded.metadata_,
                            "embedding": stmt.excluded.embedding,
                        },
                    )
                    try:
                        await session.execute(upsert_stmt)
                    except (sa.exc.ProgrammingError, sa.exc.DBAPIError) as exc:
                        msg = str(exc).lower()
                        if "no unique or exclusion constraint" in msg or "on conflict" in msg:
                            table_name = self._table_class.__tablename__
                            logger.error(
                                "IdempotentPGVectorStore: missing required unique constraint on 'node_id' in table '%s'. "
                                "Ensure Alembic Phase A migration (e8b1d9c2a3f4) is applied.",
                                table_name,
                            )
                            raise RuntimeError(
                                f"Missing required unique constraint on 'node_id' in table '{table_name}'. "
                                f"Run Alembic migration (e8b1d9c2a3f4) to establish uq_{table_name}_node_id before inserting vector nodes."
                            ) from exc
                        raise
        return ids

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Atomically upsert nodes by node_id synchronously."""
        self._initialize()
        ids: list[str] = []
        with self._session() as session:
            with session.begin():
                for node in nodes:
                    ids.append(node.node_id)
                    metadata_dict = node_to_metadata_dict(
                        node,
                        remove_text=True,
                        flat_metadata=self.flat_metadata,
                    )
                    stmt = pg_insert(self._table_class).values(
                        node_id=node.node_id,
                        embedding=node.get_embedding(),
                        text=node.get_content(metadata_mode=MetadataMode.NONE),
                        metadata_=metadata_dict,
                    )
                    upsert_stmt = stmt.on_conflict_do_update(
                        index_elements=["node_id"],
                        set_={
                            "text": stmt.excluded.text,
                            "metadata_": stmt.excluded.metadata_,
                            "embedding": stmt.excluded.embedding,
                        },
                    )
                    try:
                        session.execute(upsert_stmt)
                    except (sa.exc.ProgrammingError, sa.exc.DBAPIError) as exc:
                        msg = str(exc).lower()
                        if "no unique or exclusion constraint" in msg or "on conflict" in msg:
                            table_name = self._table_class.__tablename__
                            logger.error(
                                "IdempotentPGVectorStore: missing required unique constraint on 'node_id' in table '%s'. "
                                "Ensure Alembic Phase A migration (e8b1d9c2a3f4) is applied.",
                                table_name,
                            )
                            raise RuntimeError(
                                f"Missing required unique constraint on 'node_id' in table '{table_name}'. "
                                f"Run Alembic migration (e8b1d9c2a3f4) to establish uq_{table_name}_node_id before inserting vector nodes."
                            ) from exc
                        raise
        return ids


def get_pgvector_store(database_override: str | None = None) -> IdempotentPGVectorStore:
    """Initialize the IdempotentPGVectorStore from application settings.

    PGVectorStore uses psycopg2 (sync), so the asyncpg URL scheme is rewritten
    first.
    """
    url = settings.database_url
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://")

    parsed = urllib.parse.urlparse(url)
    target_db = database_override or parsed.path.lstrip("/")

    return IdempotentPGVectorStore.from_params(
        database=target_db,
        host=parsed.hostname or "localhost",
        password=parsed.password or "",
        port=parsed.port or 5432,
        user=parsed.username or "postgres",
        table_name=VECTOR_TABLE_NAME,
        embed_dim=768,  # For nomic-embed-text
    )
