"""Tests for Phase A: Schema, Models, and Alembic Migration Lifecycle.

Verifies:
1. LetterTypeMemory model includes source_doc_id with proper indexing.
2. DataLetterNode model matches data_letter_nodes schema.
3. Defensive data integrity checks detect NULL or duplicate node_id rows.
4. Migration upgrade and downgrade work on an isolated test database (hisho_test).
5. Existing LetterTypeMemory records are preserved without data loss.
6. The live Hisho database is never modified.
"""

import os
import urllib.parse
import pytest
from unittest.mock import MagicMock
import sqlalchemy as sa
from sqlalchemy.orm import Session

from config import settings
from db.models import Base, LetterTypeMemory, DataLetterNode
from db.migrations.versions.e8b1d9c2a3f4_phase_a_data_letter_nodes_lifecycle_and_source_doc_id import (
    check_data_integrity,
    upgrade as phase_a_upgrade,
    downgrade as phase_a_downgrade,
)


class TestModelDefinitions:
    """Unit tests for SQLAlchemy models in db/models.py."""

    def test_letter_type_memory_has_source_doc_id(self):
        """Verify LetterTypeMemory includes source_doc_id column."""
        columns = LetterTypeMemory.__table__.columns
        assert "source_doc_id" in columns
        col = columns["source_doc_id"]
        assert col.nullable is True
        assert isinstance(col.type, sa.String)
        # Check index on source_doc_id
        indices = [idx.name for idx in LetterTypeMemory.__table__.indexes]
        assert any("source_doc_id" in idx for idx in indices)

    def test_letter_type_memory_preserves_all_fields(self):
        """Verify all existing fields remain intact in LetterTypeMemory."""
        columns = LetterTypeMemory.__table__.columns
        expected_fields = [
            "id",
            "source_doc_id",
            "letter_type",
            "primary_domain",
            "description",
            "approval_chain",
            "table_info",
            "embedding",
            "template_url",
            "created_at",
        ]
        for field in expected_fields:
            assert field in columns, f"Field {field} missing from LetterTypeMemory"

    def test_data_letter_node_model_definition(self):
        """Verify DataLetterNode model definition matches vector store schema."""
        columns = DataLetterNode.__table__.columns
        assert "id" in columns
        assert "node_id" in columns
        assert "text" in columns
        assert "metadata_" in columns
        assert "embedding" in columns

        node_id_col = columns["node_id"]
        assert node_id_col.nullable is False
        assert node_id_col.unique is True

        text_col = columns["text"]
        assert text_col.nullable is False

    def test_metadata_registry_consistency(self):
        """Verify Base.metadata registers both tables consistently."""
        tables = Base.metadata.tables.keys()
        assert "letter_type_memory" in tables
        assert "data_letter_nodes" in tables


class TestCheckDataIntegrityUnit:
    """Unit tests for defensive pre-migration check_data_integrity."""

    def test_check_data_integrity_passes_when_clean(self):
        """Integrity check passes when null_count is 0 and dup_count is 0."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.scalar.side_effect = [0, 0]
        # Should not raise
        check_data_integrity(mock_conn)

    def test_check_data_integrity_fails_on_null_node_ids(self):
        """Integrity check halts migration when NULL node_ids exist."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.scalar.side_effect = [3]  # null_count = 3
        with pytest.raises(RuntimeError, match="Cannot apply NOT NULL: 3 rows have NULL node_id"):
            check_data_integrity(mock_conn)

    def test_check_data_integrity_fails_on_duplicate_node_ids(self):
        """Integrity check halts migration when duplicate node_ids exist."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.scalar.side_effect = [0, 2]  # null_count=0, dup_count=2
        with pytest.raises(RuntimeError, match="Cannot apply UNIQUE constraint: 2 duplicate node_ids exist"):
            check_data_integrity(mock_conn)


# Isolated Database Integration Tests
VEC_768 = "[" + ",".join(["0.1"] * 768) + "]"
VEC_768_ALT = "[" + ",".join(["0.2"] * 768) + "]"


def get_isolated_test_engine():
    """Build an engine pointing exclusively to the hisho_test database."""
    url = settings.database_url.replace("+asyncpg", "")
    parsed = urllib.parse.urlparse(url)
    test_url = parsed._replace(path="/hisho_test").geturl()
    try:
        engine = sa.create_engine(test_url)
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1;"))
        return engine
    except Exception:
        return None


test_engine = get_isolated_test_engine()


@pytest.mark.skipif(test_engine is None, reason="Isolated test database (hisho_test) unavailable")
class TestIsolatedDatabaseMigration:
    """Integration tests running Alembic migrations on hisho_test."""

    @classmethod
    def setup_class(cls):
        """Ensure clean slate in hisho_test."""
        with test_engine.begin() as conn:
            conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.execute(sa.text("DROP TABLE IF EXISTS data_letter_nodes CASCADE;"))
            conn.execute(sa.text("DROP TABLE IF EXISTS letter_type_memory CASCADE;"))

    def test_fresh_install_lifecycle(self):
        """Test Path A: Fresh install creates data_letter_nodes with constraints and index."""
        with test_engine.begin() as conn:
            # Create minimal letter_type_memory without source_doc_id (simulating earlier migrations)
            conn.execute(sa.text("""
                CREATE TABLE letter_type_memory (
                    id VARCHAR PRIMARY KEY,
                    letter_type VARCHAR NOT NULL,
                    primary_domain VARCHAR NOT NULL,
                    description TEXT NOT NULL,
                    approval_chain JSON NOT NULL,
                    table_info JSON,
                    embedding vector(768) NOT NULL,
                    template_url VARCHAR,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """))

            # Insert an existing pre-migration record to test preservation
            conn.execute(sa.text(f"""
                INSERT INTO letter_type_memory (id, letter_type, primary_domain, description, approval_chain, embedding)
                VALUES ('legacy_mem_1', 'FINANCIAL', 'FINANCE', 'Legacy grant', '[]', '{VEC_768}'::vector(768));
            """))

        # Run migration on connection
        from alembic.migration import MigrationContext
        from alembic.operations import Operations

        with test_engine.begin() as conn:
            ctx = MigrationContext.configure(conn)
            with Operations.context(ctx):
                phase_a_upgrade()

        # Verify data_letter_nodes schema
        inspector = sa.inspect(test_engine)
        assert inspector.has_table("data_letter_nodes")

        cols = {c["name"]: c for c in inspector.get_columns("data_letter_nodes")}
        assert cols["node_id"]["nullable"] is False

        # Verify unique constraint on node_id
        uq_constraints = [c["name"] for c in inspector.get_unique_constraints("data_letter_nodes")]
        assert "uq_data_letter_nodes_node_id" in uq_constraints

        # Verify HNSW index
        indices = [idx["name"] for idx in inspector.get_indexes("data_letter_nodes")]
        assert "data_letter_nodes_embedding_hnsw_idx" in indices

        # Verify letter_type_memory source_doc_id added
        ltm_cols = {c["name"]: c for c in inspector.get_columns("letter_type_memory")}
        assert "source_doc_id" in ltm_cols
        assert ltm_cols["source_doc_id"]["nullable"] is True

        # Verify pre-existing letter_type_memory record preserved with NULL source_doc_id
        with test_engine.connect() as conn:
            row = conn.execute(sa.text("SELECT id, letter_type, source_doc_id FROM letter_type_memory WHERE id = 'legacy_mem_1';")).mappings().first()
            assert row is not None
            assert row["letter_type"] == "FINANCIAL"
            assert row["source_doc_id"] is None

    def test_node_id_uniqueness_enforced(self):
        """Verify that UNIQUE constraint blocks duplicate node_ids."""
        with test_engine.connect() as conn:
            # First insert succeeds
            conn.execute(sa.text(f"""
                INSERT INTO data_letter_nodes (node_id, text, metadata_, embedding)
                VALUES ('doc_1#tab_1', 'Sample text', '{{}}', '{VEC_768}'::vector(768));
            """))
            conn.commit()

            # Second insert with same node_id fails with IntegrityError
            with pytest.raises(Exception) as exc_info:
                conn.execute(sa.text(f"""
                    INSERT INTO data_letter_nodes (node_id, text, metadata_, embedding)
                    VALUES ('doc_1#tab_1', 'Duplicate text', '{{}}', '{VEC_768_ALT}'::vector(768));
                """))
                conn.commit()
            assert "uq_data_letter_nodes_node_id" in str(exc_info.value) or "duplicate key" in str(exc_info.value)

    def test_node_id_not_null_enforced(self):
        """Verify that NOT NULL constraint blocks NULL node_id."""
        with test_engine.connect() as conn:
            with pytest.raises(Exception) as exc_info:
                conn.execute(sa.text(f"""
                    INSERT INTO data_letter_nodes (node_id, text, metadata_, embedding)
                    VALUES (NULL, 'Null node id text', '{{}}', '{VEC_768}'::vector(768));
                """))
                conn.commit()
            assert "null value in column \"node_id\"" in str(exc_info.value)

    def test_downgrade_reverts_cleanly(self):
        """Verify downgrade drops constraint and source_doc_id column cleanly."""
        from alembic.migration import MigrationContext
        from alembic.operations import Operations

        with test_engine.begin() as conn:
            ctx = MigrationContext.configure(conn)
            with Operations.context(ctx):
                phase_a_downgrade()

        inspector = sa.inspect(test_engine)
        # letter_type_memory should no longer have source_doc_id
        ltm_cols = [c["name"] for c in inspector.get_columns("letter_type_memory")]
        assert "source_doc_id" not in ltm_cols

        # data_letter_nodes node_id should be nullable and unique constraint dropped
        dln_cols = {c["name"]: c for c in inspector.get_columns("data_letter_nodes")}
        assert dln_cols["node_id"]["nullable"] is True
        uq_constraints = [c["name"] for c in inspector.get_unique_constraints("data_letter_nodes")]
        assert "uq_data_letter_nodes_node_id" not in uq_constraints

    def test_existing_instance_clean_upgrade(self):
        """Test Path B: Migration succeeds on existing database with valid data."""
        from alembic.migration import MigrationContext
        from alembic.operations import Operations

        # Setup existing database state (unconstrained table created dynamically)
        with test_engine.begin() as conn:
            conn.execute(sa.text("DROP TABLE IF EXISTS data_letter_nodes CASCADE;"))
            conn.execute(sa.text("DROP TABLE IF EXISTS letter_type_memory CASCADE;"))
            conn.execute(sa.text("""
                CREATE TABLE data_letter_nodes (
                    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    node_id VARCHAR,
                    text VARCHAR NOT NULL,
                    metadata_ JSON,
                    embedding vector(768)
                );
            """))
            conn.execute(sa.text(f"""
                INSERT INTO data_letter_nodes (node_id, text, metadata_, embedding)
                VALUES ('exist_1', 'Valid text 1', '{{}}', '{VEC_768}'::vector(768)),
                       ('exist_2', 'Valid text 2', '{{}}', '{VEC_768}'::vector(768));
            """))
            conn.execute(sa.text("""
                CREATE TABLE letter_type_memory (
                    id VARCHAR PRIMARY KEY,
                    letter_type VARCHAR NOT NULL,
                    primary_domain VARCHAR NOT NULL,
                    description TEXT NOT NULL,
                    approval_chain JSON NOT NULL,
                    table_info JSON,
                    embedding vector(768) NOT NULL,
                    template_url VARCHAR,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """))

        # Run upgrade on existing instance
        with test_engine.begin() as conn:
            ctx = MigrationContext.configure(conn)
            with Operations.context(ctx):
                phase_a_upgrade()

        # Verify constraints applied
        inspector = sa.inspect(test_engine)
        dln_cols = {c["name"]: c for c in inspector.get_columns("data_letter_nodes")}
        assert dln_cols["node_id"]["nullable"] is False

        uq_constraints = [c["name"] for c in inspector.get_unique_constraints("data_letter_nodes")]
        assert "uq_data_letter_nodes_node_id" in uq_constraints

        ltm_cols = [c["name"] for c in inspector.get_columns("letter_type_memory")]
        assert "source_doc_id" in ltm_cols

    def test_existing_instance_fails_on_duplicate_node_ids(self):
        """Test Path B defensive check: Fails on existing instance with duplicate node_ids."""
        from alembic.migration import MigrationContext
        from alembic.operations import Operations

        with test_engine.begin() as conn:
            conn.execute(sa.text("DROP TABLE IF EXISTS data_letter_nodes CASCADE;"))
            conn.execute(sa.text("""
                CREATE TABLE data_letter_nodes (
                    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    node_id VARCHAR,
                    text VARCHAR NOT NULL,
                    metadata_ JSON,
                    embedding vector(768)
                );
            """))
            conn.execute(sa.text(f"""
                INSERT INTO data_letter_nodes (node_id, text, metadata_, embedding)
                VALUES ('dup_id', 'Text 1', '{{}}', '{VEC_768}'::vector(768)),
                       ('dup_id', 'Text 2', '{{}}', '{VEC_768}'::vector(768));
            """))

        with test_engine.begin() as conn:
            ctx = MigrationContext.configure(conn)
            with pytest.raises(RuntimeError, match="Cannot apply UNIQUE constraint: 1 duplicate node_ids exist"):
                with Operations.context(ctx):
                    phase_a_upgrade()

    def test_existing_instance_fails_on_null_node_id(self):
        """Test Path B defensive check: Fails on existing instance with NULL node_id."""
        from alembic.migration import MigrationContext
        from alembic.operations import Operations

        with test_engine.begin() as conn:
            conn.execute(sa.text("DROP TABLE IF EXISTS data_letter_nodes CASCADE;"))
            conn.execute(sa.text("""
                CREATE TABLE data_letter_nodes (
                    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    node_id VARCHAR,
                    text VARCHAR NOT NULL,
                    metadata_ JSON,
                    embedding vector(768)
                );
            """))
            conn.execute(sa.text(f"""
                INSERT INTO data_letter_nodes (node_id, text, metadata_, embedding)
                VALUES (NULL, 'Null text', '{{}}', '{VEC_768}'::vector(768));
            """))

        with test_engine.begin() as conn:
            ctx = MigrationContext.configure(conn)
            with pytest.raises(RuntimeError, match="Cannot apply NOT NULL: 1 rows have NULL node_id"):
                with Operations.context(ctx):
                    phase_a_upgrade()
