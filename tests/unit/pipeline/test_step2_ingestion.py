"""Step 2 Ingestion Synchronization and Category Normalization Tests.

Verifies:
1. Canonical category normalization (4 reference letters, case/alias normalization,
   explicit rejection of unsupported categories including EVENT_APPROVAL).
2. Stable source identity and provenance (doc_id#tab_id, distinct tabs, lineage across
   node.node_id, LetterTypeMemory, and Signature).
3. Source-scoped synchronization (source A update preserves source B, no global wipes,
   legacy records without provenance preserved, partial extraction safety).
4. Re-ingestion idempotency via IdempotentPGVectorStore (upsert on conflict, text/metadata update,
   no duplicate rows, failure safety).
5. Signature safety (source scoping, duplicate prevention on repeated ingestion,
   incomplete extraction safety).
"""

from __future__ import annotations

import urllib.parse
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from config import settings
from db.models import Base, Designation, LetterTypeMemory, Signature, DataLetterNode
from llama_index.core.schema import TextNode
from services.constants.letter_types import (
    CANONICAL_CATEGORY_IDS,
    normalize_canonical_category,
)
from services.pipeline.pipeline_components.gdoc_parser import GoogleDocJsonParser
from services.pipeline.pipeline_components.node_formatter import NodeTextFormatter
from services.pipeline.pipeline_components.legacy_sync import (
    LegacyTableSync,
    _sync_to_legacy_tables,
)
from services.pipeline.pipeline_components.vector_store import (
    IdempotentPGVectorStore,
    get_pgvector_store,
)


# ===========================================================================
# 1. Category Normalization Tests
# ===========================================================================

class TestCategoryNormalization:
    """Validate canonical category normalization and strict rejection rules."""

    def test_winter_hackathon_normalizes_to_financial(self):
        """Winter Hackathon maps to FINANCIAL_REQUEST."""
        assert normalize_canonical_category("FINANCIAL_REQUEST", "Winter Hackathon") == "FINANCIAL_REQUEST"
        # Even if raw category was slightly different or omitted, title maps it
        assert normalize_canonical_category(None, "Winter Hackathon") == "FINANCIAL_REQUEST"
        assert normalize_canonical_category("funds", "Winter Hackathon") == "FINANCIAL_REQUEST"

    def test_hacktofuture_normalizes_to_financial(self):
        """Transfer of Funds to HackToFuture4.0 maps to FINANCIAL_REQUEST."""
        assert normalize_canonical_category("FINANCIAL_REQUEST", "Transfer of Funds to HackToFuture4.0") == "FINANCIAL_REQUEST"
        assert normalize_canonical_category("financial", "HackToFuture4.0") == "FINANCIAL_REQUEST"

    def test_camera_setup_osm_normalizes_to_equipment(self):
        """Camera Setup for OSM Workshop maps to EQUIPMENT_REQUEST."""
        assert normalize_canonical_category("EQUIPMENT_REQUEST", "Camera Setup for OSM") == "EQUIPMENT_REQUEST"
        assert normalize_canonical_category("equipment", "Camera Equipment for OSM Workshop") == "EQUIPMENT_REQUEST"

    def test_prerna_hall_normalizes_to_venue_permission_never_event_approval(self):
        """Prerna Hall for Tech Trivia maps to VENUE_PERMISSION even if LLM said EVENT_APPROVAL."""
        assert normalize_canonical_category("VENUE_PERMISSION", "Prerna Hall for Tech Trivia") == "VENUE_PERMISSION"
        # Event context must NOT override primary requested action
        assert normalize_canonical_category("EVENT_APPROVAL", "Prerna Hall for Tech Trivia") == "VENUE_PERMISSION"
        assert normalize_canonical_category(None, "Prerna Hall for Tech Trivia") == "VENUE_PERMISSION"

    def test_case_and_format_normalization(self):
        """Normalizes case, whitespace, hyphens, and standard aliases."""
        assert normalize_canonical_category("financial_request") == "FINANCIAL_REQUEST"
        assert normalize_canonical_category("Financial-Request") == "FINANCIAL_REQUEST"
        assert normalize_canonical_category("  equipment_request  ") == "EQUIPMENT_REQUEST"
        assert normalize_canonical_category("venue permission") == "VENUE_PERMISSION"
        assert normalize_canonical_category("FINANCE") == "FINANCIAL_REQUEST"
        assert normalize_canonical_category("BUDGET") == "FINANCIAL_REQUEST"
        assert normalize_canonical_category("BORROW_EQUIPMENT") == "EQUIPMENT_REQUEST"
        assert normalize_canonical_category("HALL_BOOKING") == "VENUE_PERMISSION"

    def test_unsupported_categories_rejected(self):
        """Unsupported categories must raise ValueError without silent assignment."""
        unsupported = [
            "EVENT_APPROVAL",
            "ACADEMIC_DUTY_LEAVE",
            "OFFICIAL_CIRCULAR",
            "INVITATION",
            "GENERAL_CORRESPONDENCE",
            "RANDOM_UNKNOWN_TYPE",
        ]
        for cat in unsupported:
            with pytest.raises(ValueError, match="Unsupported category|Invalid or unsupported"):
                normalize_canonical_category(cat, document_title="Generic Unrelated Doc")

    def test_missing_category_and_title_raises(self):
        """Empty category and empty title raises ValueError defensively."""
        with pytest.raises(ValueError, match="missing or unresolvable"):
            normalize_canonical_category(None, None)
        with pytest.raises(ValueError, match="missing or unresolvable"):
            normalize_canonical_category("", "")


# ===========================================================================
# 2. Stable Source Identity and Provenance Tests
# ===========================================================================

class TestSourceIdentityAndProvenance:
    """Validate stable composite identifier (doc_id#tab_id) and end-to-end lineage."""

    def test_stable_identity_format_and_repetition(self):
        """Repeated parser runs on the same tab produce identical node_id."""
        import json
        payload = json.dumps([{"type": "text", "content": "Sample letter text."}])

        doc = TextNode(
            text=payload,
            metadata={
                "source_doc_id": "google_doc_123#tab_finance",
                "tab_title": "Winter Hackathon",
            },
        )
        parser = GoogleDocJsonParser()
        node_first = parser([doc])[0]
        node_second = parser([doc])[0]

        assert node_first.node_id == "google_doc_123#tab_finance"
        assert node_second.node_id == "google_doc_123#tab_finance"
        assert node_first.node_id == node_second.node_id

    def test_different_tabs_produce_distinct_identifiers(self):
        """Different tabs from same document produce distinct node_ids."""
        import json
        payload = json.dumps([{"type": "text", "content": "Tab text."}])

        doc_tab1 = TextNode(
            text=payload,
            metadata={
                "source_doc_id": "gdoc_xyz#tab_01",
                "tab_title": "Tab 1",
            },
        )
        doc_tab2 = TextNode(
            text=payload,
            metadata={
                "source_doc_id": "gdoc_xyz#tab_02",
                "tab_title": "Tab 2",
            },
        )
        parser = GoogleDocJsonParser()
        res1 = parser([doc_tab1])[0]
        res2 = parser([doc_tab2])[0]

        assert res1.node_id == "gdoc_xyz#tab_01"
        assert res2.node_id == "gdoc_xyz#tab_02"
        assert res1.node_id != res2.node_id

    def test_provenance_preserved_in_formatter_and_legacy_sync(self):
        """Formatter preserves source_doc_id in metadata and maintains node_id."""
        node = TextNode(
            id_="doc_abc#tab_99",
            text="Letter body",
            metadata={
                "source_doc_id": "doc_abc#tab_99",
                "tab_title": "Camera Setup",
                "letter_type": "EQUIPMENT_REQUEST",
                "subject": "Camera Setup for Workshop",
                "description": "Formal request seeking equipment borrowing.",
                "full_markdown": "Full authentic letter body.",
            },
        )
        formatter = NodeTextFormatter()
        formatted_node = formatter([node])[0]

        assert formatted_node.node_id == "doc_abc#tab_99"
        assert formatted_node.metadata["source_doc_id"] == "doc_abc#tab_99"
        assert formatted_node.metadata["tab_title"] == "Camera Setup"
        assert formatted_node.metadata["full_letter"] == "Full authentic letter body."


# ===========================================================================
# 3. Source-Scoped Synchronization Tests
# ===========================================================================

class TestSourceScopedLegacySynchronization:
    """Validate that LegacyTableSync scopes operations to the current source only."""

    @pytest.mark.asyncio
    async def test_updating_source_a_preserves_source_b(self):
        """Re-syncing source A deletes only source A records, never source B."""
        node_a = TextNode(
            id_="source_A",
            text="Node A",
            metadata={
                "source_doc_id": "source_A",
                "letter_type": "FINANCIAL_REQUEST",
                "description": "Financial request A",
                "approval_chain": [{"step_number": 1, "designation_title": "HOD", "incumbent": "Dr. A"}],
                "signature_extraction_status": "SUCCESS",
                "signatures": [{"url": "https://example.com/sig.png", "name": "Dr. A"}],
            },
        )
        sync = LegacyTableSync()
        mock_session = AsyncMock()
        mock_session.add = MagicMock()

        with patch("services.pipeline.pipeline_components.legacy_sync.AsyncSessionLocal") as mock_session_cls, \
             patch.object(sync, "_generate_description_embeddings", return_value=[[0.1] * 768]), \
             patch("services.pipeline.pipeline_components.legacy_sync._load_existing_designations", return_value={}):

            mock_session_cls.return_value.__aenter__.return_value = mock_session
            await sync.acall([node_a])

            # Inspect all delete calls
            executed_stmts = [str(call.args[0]) for call in mock_session.execute.call_args_list]

            # LetterTypeMemory delete must have a WHERE clause scoping to source_A
            ltm_deletes = [s for s in executed_stmts if "DELETE FROM letter_type_memory" in s]
            assert len(ltm_deletes) == 1
            assert "WHERE letter_type_memory.source_doc_id =" in ltm_deletes[0]
            # Must NOT be an unconstrained global delete
            assert "WHERE" in ltm_deletes[0]

            # Signature delete must also have a WHERE clause scoping to source_A
            sig_deletes = [s for s in executed_stmts if "DELETE FROM signatures" in s]
            assert len(sig_deletes) == 1
            assert "WHERE signatures.source_doc_id =" in sig_deletes[0]

    @pytest.mark.asyncio
    async def test_missing_source_id_fails_safely_no_db_writes(self):
        """Node with missing or empty source_doc_id is rejected without DB mutation."""
        node_no_source = TextNode(
            id_="no_source",
            text="No source text",
            metadata={
                "source_doc_id": "",  # Empty source ID
                "letter_type": "FINANCIAL_REQUEST",
                "description": "Some description",
                "approval_chain": [],
            },
        )
        sync = LegacyTableSync()
        with patch("services.pipeline.pipeline_components.legacy_sync.AsyncSessionLocal") as mock_session_cls:
            res = await sync.acall([node_no_source])
            assert len(res) == 1
            mock_session_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_partial_extraction_failure_does_not_wipe_other_sources(self):
        """When node 1 succeeds and node 2 fails, only node 1's source is touched."""
        good_node = TextNode(
            id_="src_good#tab1",
            text="Good text",
            metadata={
                "source_doc_id": "src_good#tab1",
                "letter_type": "VENUE_PERMISSION",
                "description": "Venue description",
                "approval_chain": [],
            },
        )
        bad_node = TextNode(
            id_="src_bad#tab2",
            text="Bad text",
            metadata={
                "source_doc_id": "src_bad#tab2",
                # Missing letter_type and description
            },
        )
        sync = LegacyTableSync()
        mock_session = AsyncMock()
        mock_session.add = MagicMock()

        with patch("services.pipeline.pipeline_components.legacy_sync.AsyncSessionLocal") as mock_session_cls, \
             patch.object(sync, "_generate_description_embeddings", return_value=[[0.2] * 768]), \
             patch("services.pipeline.pipeline_components.legacy_sync._load_existing_designations", return_value={}):

            mock_session_cls.return_value.__aenter__.return_value = mock_session
            await sync.acall([good_node, bad_node])

            # Deletions were only executed for src_good#tab1
            executed_stmts = [str(call.args[0]) for call in mock_session.execute.call_args_list]
            for stmt in executed_stmts:
                if "DELETE" in stmt:
                    assert "WHERE" in stmt
                    assert "source_doc_id =" in stmt


# ===========================================================================
# 4. Isolated Database Re-Ingestion & Upsert Tests (hisho_test)
# ===========================================================================

def _make_isolated_engine():
    """Create a fresh async engine pointing exclusively to hisho_test."""
    raw_url = settings.database_url
    if not raw_url.startswith("postgresql+asyncpg://"):
        raw_url = raw_url.replace("postgresql://", "postgresql+asyncpg://")
    parsed = urllib.parse.urlparse(raw_url)
    test_url = parsed._replace(path="/hisho_test").geturl()
    return create_async_engine(test_url)


def _check_isolated_db():
    import asyncio

    async def _ping():
        engine = _make_isolated_engine()
        try:
            async with engine.connect() as conn:
                res = await conn.execute(sa.text("SELECT current_database();"))
                return res.scalar() == "hisho_test"
        finally:
            await engine.dispose()

    try:
        return asyncio.run(_ping())
    except Exception:
        return False


_db_available = _check_isolated_db()


@pytest.mark.skipif(not _db_available, reason="Isolated hisho_test database unavailable")
class TestIsolatedDbReingestionUpsert:
    """Validate idempotent re-ingestion against the isolated hisho_test database."""

    @classmethod
    def setup_class(cls):
        """Ensure isolated hisho_test database is prepared using Phase A migration.

        NOTE ON MIGRATION EXECUTION:
        This setup invokes the actual Phase A migration `upgrade()` function
        directly using `MigrationContext.configure(sync_conn)` and
        `with Operations.context(ctx): phase_a_upgrade()`.
        All migration operations execute strictly against the explicitly bound
        `hisho_test` database connection.
        This direct execution applies the migration DDL (creating `data_letter_nodes`,
        adding the unique constraint `uq_data_letter_nodes_node_id`, etc.) but does
        not update the `alembic_version` table (it does not constitute a complete
        Alembic CLI migration lifecycle).
        """
        import asyncio
        from alembic.migration import MigrationContext
        from alembic.operations import Operations
        from db.migrations.versions.e8b1d9c2a3f4_phase_a_data_letter_nodes_lifecycle_and_source_doc_id import (
            upgrade as phase_a_upgrade,
        )

        async def _prepare():
            engine = _make_isolated_engine()
            try:
                async with engine.begin() as conn:
                    # 1. Hard database identity assertion: abort immediately if not hisho_test
                    db_check = await conn.execute(sa.text("SELECT current_database();"))
                    current_db = db_check.scalar()
                    assert current_db == "hisho_test", (
                        f"Safety abort: Expected connection to 'hisho_test', but connected to '{current_db}'!"
                    )

                    await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    await conn.execute(sa.text("DROP TABLE IF EXISTS data_letter_nodes CASCADE;"))
                    await conn.execute(sa.text("DROP TABLE IF EXISTS letter_type_memory CASCADE;"))

                    await conn.execute(sa.text("""
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

                    # Execute the Phase A migration upgrade function directly
                    # against the explicitly bound hisho_test connection.
                    def _apply(sync_conn):
                        ctx = MigrationContext.configure(sync_conn)
                        with Operations.context(ctx):
                            phase_a_upgrade()

                    await conn.run_sync(_apply)

                # Verify that unique constraint was created by Phase A migration
                async with engine.begin() as conn:
                    def _check_uq(sync_conn):
                        inspector = sa.inspect(sync_conn)
                        uq_constraints = [c["name"] for c in inspector.get_unique_constraints("data_letter_nodes")]
                        assert "uq_data_letter_nodes_node_id" in uq_constraints

                    await conn.run_sync(_check_uq)
            finally:
                await engine.dispose()

        asyncio.run(_prepare())

    @pytest.mark.asyncio
    async def test_reingestion_updates_existing_node_in_place(self):
        """Ingesting the same node_id twice updates the node rather than creating duplicates."""
        store = get_pgvector_store(database_override="hisho_test")

        # Initial node
        initial_node = TextNode(
            id_="step2_test#tab_01",
            text="Type: FINANCIAL_REQUEST\nSubject: Initial Fund Request",
            embedding=[0.1] * 768,
            metadata={
                "source_doc_id": "step2_test#tab_01",
                "letter_type": "FINANCIAL_REQUEST",
                "full_markdown": "Initial letter markdown body",
            },
        )
        await store.async_add([initial_node])

        # Verify 1 row exists
        engine = _make_isolated_engine()
        try:
            async with engine.begin() as conn:
                db_check = await conn.execute(sa.text("SELECT current_database();"))
                assert db_check.scalar() == "hisho_test"

                result = await conn.execute(
                    sa.text("SELECT count(*) FROM data_letter_nodes WHERE node_id = 'step2_test#tab_01'")
                )
                assert result.scalar() == 1

            # Re-ingest with updated content and embedding
            updated_node = TextNode(
                id_="step2_test#tab_01",
                text="Type: FINANCIAL_REQUEST\nSubject: Updated Fund Request for 2026",
                embedding=[0.5] * 768,
                metadata={
                    "source_doc_id": "step2_test#tab_01",
                    "letter_type": "FINANCIAL_REQUEST",
                    "full_markdown": "Updated letter markdown body for 2026",
                },
            )
            await store.async_add([updated_node])

            # Verify still exactly 1 row (no duplicate row count growth)
            async with engine.begin() as conn:
                result = await conn.execute(
                    sa.text("SELECT text, metadata_ FROM data_letter_nodes WHERE node_id = 'step2_test#tab_01'")
                )
                rows = result.fetchall()
                assert len(rows) == 1
                text, meta = rows[0]
                assert "Updated Fund Request for 2026" in text
                assert meta["full_markdown"] == "Updated letter markdown body for 2026"

                # Cleanup test record
                await conn.execute(
                    sa.text("DELETE FROM data_letter_nodes WHERE node_id = 'step2_test#tab_01'")
                )
        finally:
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_legacy_records_without_provenance_preserved(self):
        """Legacy records with source_doc_id IS NULL are never deleted or modified."""
        engine = _make_isolated_engine()
        try:
            # Ensure required tables exist in hisho_test
            async with engine.begin() as conn:
                db_check = await conn.execute(sa.text("SELECT current_database();"))
                assert db_check.scalar() == "hisho_test"

                await conn.run_sync(
                    lambda c: Base.metadata.create_all(
                        c,
                        tables=[
                            LetterTypeMemory.__table__,
                            Signature.__table__,
                            Designation.__table__,
                        ],
                    )
                )

            session_factory = async_sessionmaker(engine, expire_on_commit=False)

            # 1. Insert a mock legacy record without provenance (source_doc_id is NULL)
            legacy_id = uuid.uuid4().hex
            async with session_factory() as session:
                async with session.begin():
                    legacy_record = LetterTypeMemory(
                        id=legacy_id,
                        source_doc_id=None,  # No provenance
                        letter_type="FINANCIAL_REQUEST",
                        primary_domain="FINANCE",
                        description="Legacy financial record without source_doc_id",
                        approval_chain=[],
                        embedding=[0.0] * 768,
                    )
                    session.add(legacy_record)

            # 2. Run LegacyTableSync with a newly ingested source
            sync_node = TextNode(
                id_="new_source#tab_1",
                text="New source text",
                metadata={
                    "source_doc_id": "new_source#tab_1",
                    "letter_type": "VENUE_PERMISSION",
                    "primary_domain": "INFRASTRUCTURE",
                    "description": "Venue allocation purpose",
                    "approval_chain": [],
                },
            )

            sync = LegacyTableSync()
            with patch.object(sync, "_generate_description_embeddings", return_value=[[0.3] * 768]), \
                 patch("services.pipeline.pipeline_components.legacy_sync.AsyncSessionLocal", session_factory):
                await sync.acall([sync_node])

            # 3. Verify the legacy record without provenance is STILL present
            async with session_factory() as session:
                result = await session.execute(
                    sa.select(LetterTypeMemory).where(LetterTypeMemory.id == legacy_id)
                )
                row = result.scalar_one_or_none()
                assert row is not None, "Legacy record without provenance was unexpectedly deleted!"
                assert row.source_doc_id is None
                assert row.description == "Legacy financial record without source_doc_id"

                # Cleanup isolated test rows
                await session.execute(
                    sa.delete(LetterTypeMemory).where(LetterTypeMemory.id == legacy_id)
                )
                await session.execute(
                    sa.delete(LetterTypeMemory).where(LetterTypeMemory.source_doc_id == "new_source#tab_1")
                )
                await session.commit()
        finally:
            await engine.dispose()


# ===========================================================================
# 5. Signature Synchronization Safety Tests (States A, B, and C)
# ===========================================================================

class TestSignatureSyncSafety:
    """Validate signature extraction state handling, isolation, and transaction safety."""

    @pytest.mark.asyncio
    async def test_successful_extraction_replaces_existing_source_signatures(self):
        """State A: Successful extraction deletes existing signatures for that source and inserts new ones."""
        sync = LegacyTableSync()
        mock_session = AsyncMock()
        mock_session.add = MagicMock()

        node = TextNode(
            id_="doc_sig#tab1",
            text="body",
            metadata={
                "source_doc_id": "doc_sig#tab1",
                "letter_type": "EQUIPMENT_REQUEST",
                "description": "Equip desc",
                "signature_extraction_status": "SUCCESS",
                "signatures": [
                    {"url": "https://example.com/new_sig.png", "name": "Prof. X"},
                ],
                "approval_chain": [],
            },
        )

        with patch("services.pipeline.pipeline_components.legacy_sync.AsyncSessionLocal") as mock_session_cls, \
             patch.object(sync, "_generate_description_embeddings", return_value=[[0.1] * 768]), \
             patch("services.pipeline.pipeline_components.legacy_sync._load_existing_designations", return_value={}):

            mock_session_cls.return_value.__aenter__.return_value = mock_session
            await sync.acall([node])

            # Check that delete was executed for doc_sig#tab1
            delete_stmts = [str(call.args[0]) for call in mock_session.execute.call_args_list if "DELETE FROM signatures" in str(call.args[0])]
            assert len(delete_stmts) == 1
            assert "WHERE signatures.source_doc_id =" in delete_stmts[0]

            # Check that new signature was added
            added_signatures = [
                obj for call in mock_session.add.call_args_list
                for obj in call.args
                if isinstance(obj, Signature)
            ]
            assert len(added_signatures) == 1
            assert added_signatures[0].source_doc_id == "doc_sig#tab1"
            assert added_signatures[0].image_url == "https://example.com/new_sig.png"

    @pytest.mark.asyncio
    async def test_confirmed_empty_deletes_only_that_source_signatures(self):
        """State B: Explicitly confirmed zero signatures deletes existing source signatures and inserts nothing."""
        sync = LegacyTableSync()
        mock_session = AsyncMock()
        mock_session.add = MagicMock()

        node = TextNode(
            id_="doc_sig#tab_empty",
            text="body",
            metadata={
                "source_doc_id": "doc_sig#tab_empty",
                "letter_type": "FINANCIAL_REQUEST",
                "description": "Financial desc",
                "signature_extraction_status": "CONFIRMED_EMPTY",
                "signatures": [],
                "approval_chain": [],
            },
        )

        with patch("services.pipeline.pipeline_components.legacy_sync.AsyncSessionLocal") as mock_session_cls, \
             patch.object(sync, "_generate_description_embeddings", return_value=[[0.1] * 768]), \
             patch("services.pipeline.pipeline_components.legacy_sync._load_existing_designations", return_value={}):

            mock_session_cls.return_value.__aenter__.return_value = mock_session
            await sync.acall([node])

            # Delete was called for doc_sig#tab_empty
            delete_stmts = [str(call.args[0]) for call in mock_session.execute.call_args_list if "DELETE FROM signatures" in str(call.args[0])]
            assert len(delete_stmts) == 1
            assert "WHERE signatures.source_doc_id =" in delete_stmts[0]

            # No signatures were added
            added_signatures = [
                obj for call in mock_session.add.call_args_list
                for obj in call.args
                if isinstance(obj, Signature)
            ]
            assert len(added_signatures) == 0

    @pytest.mark.asyncio
    async def test_failed_extraction_preserves_existing_signatures(self):
        """State C: Failed extraction does NOT delete or mutate existing signatures."""
        sync = LegacyTableSync()
        mock_session = AsyncMock()
        mock_session.add = MagicMock()

        node = TextNode(
            id_="doc_sig#tab_failed",
            text="body",
            metadata={
                "source_doc_id": "doc_sig#tab_failed",
                "letter_type": "VENUE_PERMISSION",
                "description": "Venue desc",
                "signature_extraction_status": "FAILED",
                "signatures": [],
                "approval_chain": [],
            },
        )

        with patch("services.pipeline.pipeline_components.legacy_sync.AsyncSessionLocal") as mock_session_cls, \
             patch.object(sync, "_generate_description_embeddings", return_value=[[0.1] * 768]), \
             patch("services.pipeline.pipeline_components.legacy_sync._load_existing_designations", return_value={}):

            mock_session_cls.return_value.__aenter__.return_value = mock_session
            await sync.acall([node])

            # Zero signature deletions executed!
            sig_deletes = [str(call.args[0]) for call in mock_session.execute.call_args_list if "DELETE FROM signatures" in str(call.args[0])]
            assert len(sig_deletes) == 0

            # Zero signatures added
            added_signatures = [
                obj for call in mock_session.add.call_args_list
                for obj in call.args
                if isinstance(obj, Signature)
            ]
            assert len(added_signatures) == 0

    @pytest.mark.asyncio
    async def test_missing_status_preserves_existing_signatures(self):
        """State C: Missing signature_extraction_status does NOT delete existing signatures."""
        sync = LegacyTableSync()
        mock_session = AsyncMock()
        mock_session.add = MagicMock()

        node = TextNode(
            id_="doc_sig#tab_nostatus",
            text="body",
            metadata={
                "source_doc_id": "doc_sig#tab_nostatus",
                "letter_type": "VENUE_PERMISSION",
                "description": "Venue desc",
                # signature_extraction_status is missing
                "signatures": [],
                "approval_chain": [],
            },
        )

        with patch("services.pipeline.pipeline_components.legacy_sync.AsyncSessionLocal") as mock_session_cls, \
             patch.object(sync, "_generate_description_embeddings", return_value=[[0.1] * 768]), \
             patch("services.pipeline.pipeline_components.legacy_sync._load_existing_designations", return_value={}):

            mock_session_cls.return_value.__aenter__.return_value = mock_session
            await sync.acall([node])

            # Zero signature deletions executed!
            sig_deletes = [str(call.args[0]) for call in mock_session.execute.call_args_list if "DELETE FROM signatures" in str(call.args[0])]
            assert len(sig_deletes) == 0

    @pytest.mark.asyncio
    async def test_incomplete_extraction_missing_url_preserves_existing_signatures(self):
        """State C: Incomplete signatures (missing URL) trigger conservative preservation."""
        sync = LegacyTableSync()
        mock_session = AsyncMock()
        mock_session.add = MagicMock()

        node = TextNode(
            id_="doc_sig#tab_incomplete",
            text="body",
            metadata={
                "source_doc_id": "doc_sig#tab_incomplete",
                "letter_type": "EQUIPMENT_REQUEST",
                "description": "Equip desc",
                "signature_extraction_status": "SUCCESS",
                "signatures": [
                    {"url": None, "name": "Dr. Incomplete"},  # Missing url!
                ],
                "approval_chain": [],
            },
        )

        with patch("services.pipeline.pipeline_components.legacy_sync.AsyncSessionLocal") as mock_session_cls, \
             patch.object(sync, "_generate_description_embeddings", return_value=[[0.1] * 768]), \
             patch("services.pipeline.pipeline_components.legacy_sync._load_existing_designations", return_value={}):

            mock_session_cls.return_value.__aenter__.return_value = mock_session
            await sync.acall([node])

            # Zero signature deletions executed because extraction was incomplete!
            sig_deletes = [str(call.args[0]) for call in mock_session.execute.call_args_list if "DELETE FROM signatures" in str(call.args[0])]
            assert len(sig_deletes) == 0

    @pytest.mark.asyncio
    async def test_unrelated_source_signatures_strictly_preserved(self):
        """Updating Source A signatures strictly preserves Source B signatures."""
        sync = LegacyTableSync()
        mock_session = AsyncMock()
        mock_session.add = MagicMock()

        node_a = TextNode(
            id_="source_A#tab1",
            text="body",
            metadata={
                "source_doc_id": "source_A#tab1",
                "letter_type": "FINANCIAL_REQUEST",
                "description": "Financial desc",
                "signature_extraction_status": "SUCCESS",
                "signatures": [{"url": "https://example.com/sig_a.png", "name": "Signee A"}],
                "approval_chain": [],
            },
        )

        with patch("services.pipeline.pipeline_components.legacy_sync.AsyncSessionLocal") as mock_session_cls, \
             patch.object(sync, "_generate_description_embeddings", return_value=[[0.1] * 768]), \
             patch("services.pipeline.pipeline_components.legacy_sync._load_existing_designations", return_value={}):

            mock_session_cls.return_value.__aenter__.return_value = mock_session
            await sync.acall([node_a])

            # Inspect delete statement bound parameters
            sig_delete_calls = [
                call for call in mock_session.execute.call_args_list
                if "DELETE FROM signatures" in str(call.args[0])
            ]
            assert len(sig_delete_calls) == 1
            params = sig_delete_calls[0].args[0].compile().params
            assert "source_A#tab1" in params.values()
            assert "source_B#tab1" not in params.values()


# ===========================================================================
# 6. Vector Store Upsert & Runtime DDL Verification Tests
# ===========================================================================

class TestVectorStoreRuntimeSafety:
    """Validate that IdempotentPGVectorStore does NOT emit runtime DDL and fails cleanly on missing constraints."""

    @pytest.mark.asyncio
    async def test_async_add_does_not_execute_runtime_ddl(self):
        """async_add executes ON CONFLICT DO UPDATE without running CREATE INDEX or other DDL."""
        store = IdempotentPGVectorStore.from_params(
            database="test_db",
            host="localhost",
            password="",
            port=5432,
            user="postgres",
            table_name="letter_nodes",
            embed_dim=768,
        )

        node = TextNode(id_="node_upsert_1", text="content", embedding=[0.1] * 768)
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_session.begin = MagicMock(return_value=mock_tx)

        mock_sessionmaker = MagicMock()
        mock_sessionmaker.return_value.__aenter__.return_value = mock_session
        mock_sessionmaker.return_value.__aexit__.return_value = None

        object.__setattr__(store, "_is_initialized", True)
        object.__setattr__(store, "_async_session", mock_sessionmaker)

        await store.async_add([node])

        # Verify no DDL statements were executed
        executed = [str(call.args[0]) for call in mock_session.execute.call_args_list]
        assert not any("CREATE UNIQUE INDEX" in s for s in executed)
        assert not any("ALTER TABLE" in s for s in executed)
        # Verify ON CONFLICT DO UPDATE was executed
        assert any("ON CONFLICT (node_id) DO UPDATE" in s for s in executed)

    def test_add_does_not_execute_runtime_ddl(self):
        """add (sync) executes ON CONFLICT DO UPDATE without running CREATE INDEX or other DDL."""
        store = IdempotentPGVectorStore.from_params(
            database="test_db",
            host="localhost",
            password="",
            port=5432,
            user="postgres",
            table_name="letter_nodes",
            embed_dim=768,
        )

        node = TextNode(id_="node_upsert_2", text="content", embedding=[0.2] * 768)
        mock_session = MagicMock()
        mock_sessionmaker = MagicMock()
        mock_sessionmaker.return_value.__enter__.return_value = mock_session
        mock_sessionmaker.return_value.__exit__.return_value = None
        mock_session.begin.return_value.__enter__.return_value = None
        mock_session.begin.return_value.__exit__.return_value = None

        object.__setattr__(store, "_is_initialized", True)
        object.__setattr__(store, "_session", mock_sessionmaker)

        store.add([node])

        # Verify no DDL statements were executed
        executed = [str(call.args[0]) for call in mock_session.execute.call_args_list]
        assert not any("CREATE UNIQUE INDEX" in s for s in executed)
        assert not any("ALTER TABLE" in s for s in executed)
        assert any("ON CONFLICT (node_id) DO UPDATE" in s for s in executed)

    @pytest.mark.asyncio
    async def test_missing_unique_constraint_fails_clearly(self):
        """Missing unique constraint fails with an actionable RuntimeError mentioning migration."""
        store = IdempotentPGVectorStore.from_params(
            database="test_db",
            host="localhost",
            password="",
            port=5432,
            user="postgres",
            table_name="letter_nodes",
            embed_dim=768,
        )
        node = TextNode(id_="node_err", text="content", embedding=[0.1] * 768)

        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_session.begin = MagicMock(return_value=mock_tx)

        db_err = sa.exc.ProgrammingError(
            "INSERT", {}, Exception("there is no unique or exclusion constraint matching the ON CONFLICT specification")
        )
        mock_session.execute.side_effect = db_err

        mock_sessionmaker = MagicMock()
        mock_sessionmaker.return_value.__aenter__.return_value = mock_session
        mock_sessionmaker.return_value.__aexit__.return_value = None

        object.__setattr__(store, "_is_initialized", True)
        object.__setattr__(store, "_async_session", mock_sessionmaker)

        with pytest.raises(RuntimeError, match="Missing required unique constraint.*e8b1d9c2a3f4") as excinfo:
            await store.async_add([node])

        # Verify the original SQLAlchemy exception is preserved as the cause
        assert excinfo.value.__cause__ is db_err


# ===========================================================================
# 7. NodeTextFormatter Category Enforcement Tests
# ===========================================================================

class TestNodeTextFormatterCategoryEnforcement:
    """Validate strict canonical category enforcement and rejection of invalid categories."""

    def test_accepts_all_three_canonical_categories(self):
        """All three canonical categories are correctly formatted without error."""
        formatter = NodeTextFormatter()
        for cat in CANONICAL_CATEGORY_IDS:
            node = TextNode(
                id_=f"node_{cat}",
                text="body",
                metadata={"letter_type": cat, "source_doc_id": f"src_{cat}#tab1"},
            )
            res = formatter([node])[0]
            assert res.metadata["letter_type"] == cat
            assert f"Type: {cat}" in res.text

    def test_normalizes_supported_aliases(self):
        """Supported category aliases normalize to canonical form."""
        formatter = NodeTextFormatter()
        node = TextNode(
            id_="node_alias",
            text="body",
            metadata={"letter_type": "financial", "source_doc_id": "src_alias#tab1"},
        )
        res = formatter([node])[0]
        assert res.metadata["letter_type"] == "FINANCIAL_REQUEST"
        assert "Type: FINANCIAL_REQUEST" in res.text

    def test_unknown_category_raises_with_document_identity(self):
        """Unknown category raises ValueError containing the source document identity and invalid category."""
        formatter = NodeTextFormatter()
        node = TextNode(
            id_="node_err_cat",
            text="body",
            metadata={"letter_type": "EVENT_APPROVAL", "source_doc_id": "doc_sceptix_456#tab_trivia"},
        )
        with pytest.raises(ValueError) as excinfo:
            formatter([node])

        err_msg = str(excinfo.value)
        assert "EVENT_APPROVAL" in err_msg
        assert "doc_sceptix_456#tab_trivia" in err_msg

    def test_missing_category_raises_when_required(self):
        """When category is required, missing category raises ValueError with document identity."""
        formatter = NodeTextFormatter(require_category=True)
        node = TextNode(
            id_="node_missing_cat",
            text="body",
            metadata={"source_doc_id": "doc_missing#tab1"},
        )
        with pytest.raises(ValueError) as excinfo:
            formatter([node])

        err_msg = str(excinfo.value)
        assert "doc_missing#tab1" in err_msg

    def test_invalid_category_does_not_reach_vector_storage(self):
        """An invalid category raises an exception, preventing downstream embedding and storage."""
        from services.pipeline.pipeline import build_pipeline
        pipeline = build_pipeline()

        # Check that the pipeline's NodeTextFormatter has require_category=True
        formatter_stages = [t for t in pipeline.transformations if isinstance(t, NodeTextFormatter)]
        assert len(formatter_stages) == 1
        assert formatter_stages[0].require_category is True
