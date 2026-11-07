"""Unit and isolated integration tests for Sceptix seed configuration."""

import pytest
import urllib.parse
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from config import settings
from db.models import Base, LetterType, Official
from scripts.seed_sceptix_metadata import (
    SCEPTIX_LETTER_TYPES,
    seed_sceptix_letter_types,
    run_seed,
)


class TestSceptixSeedDataStructure:
    """Validate structure and integrity of Sceptix seed metadata in memory."""

    def test_seed_contains_exactly_three_canonical_types(self):
        """The seed list must define exactly the three canonical categories."""
        expected_ids = {"FINANCIAL_REQUEST", "EQUIPMENT_REQUEST", "VENUE_PERMISSION"}
        actual_ids = {item["id"] for item in SCEPTIX_LETTER_TYPES}
        assert actual_ids == expected_ids
        assert len(SCEPTIX_LETTER_TYPES) == 3

    def test_approval_chains_have_valid_structure(self):
        """Each approval chain step must have step, role, and action without mock personnel."""
        for item in SCEPTIX_LETTER_TYPES:
            chain = item["approval_chain"]
            assert isinstance(chain, list)
            assert len(chain) >= 2, f"{item['id']} approval chain should have at least 2 steps"

            steps = [s["step"] for s in chain]
            assert steps == list(range(1, len(chain) + 1)), "Steps must be sequentially numbered from 1"

            for step in chain:
                assert "step" in step and isinstance(step["step"], int)
                assert "role" in step and isinstance(step["role"], str) and len(step["role"]) > 0
                assert "action" in step and isinstance(step["action"], str) and len(step["action"]) > 0
                # Must specify institutional designations, never personal names
                assert not any(name in step["role"] for name in ["Dr.", "Mr.", "Ms.", "Prof."])

    def test_templates_are_non_empty(self):
        """Templates must provide institutional markdown guidance."""
        for item in SCEPTIX_LETTER_TYPES:
            assert "template" in item and len(item["template"]) > 50

    def test_unconfirmed_run_seed_raises(self):
        """Calling run_seed without explicit confirmation must fail defensively."""
        with pytest.raises(RuntimeError, match="requires explicit confirmation"):
            import asyncio
            asyncio.run(run_seed(confirm=False, overwrite=False))

    def test_equipment_approval_chain_has_no_hod_stage(self):
        """Equipment request seed chain must not include an HOD stage.

        Historical Camera Setup for OSM routes directly to CCC without HOD.
        """
        equipment = next(item for item in SCEPTIX_LETTER_TYPES if item["id"] == "EQUIPMENT_REQUEST")
        roles = [step["role"] for step in equipment["approval_chain"]]
        for role in roles:
            assert "HOD" not in role.upper(), (
                f"Equipment approval chain must not contain HOD stage; found {role!r}"
            )
            assert "Head of Department" not in role, (
                f"Equipment approval chain must not contain HOD stage; found {role!r}"
            )

    def test_venue_permission_approval_chain_recipient_is_assistant_director(self):
        """Venue permission approval chain must designate 'Assistant Director' as final step.

        Must not use blended 'Assistant Director / Principal' designation.
        """
        venue = next(item for item in SCEPTIX_LETTER_TYPES if item["id"] == "VENUE_PERMISSION")
        final_step = venue["approval_chain"][-1]
        assert "Assistant Director" in final_step["role"], (
            f"Final approval step role must include 'Assistant Director', got {final_step['role']!r}"
        )
        assert "Principal" not in final_step["role"], (
            f"Final approval step must not blend 'Principal' into 'Assistant Director' role"
        )

    def test_no_event_approval_in_seed(self):
        """EVENT_APPROVAL must not appear in the seed data."""
        ids = {item["id"] for item in SCEPTIX_LETTER_TYPES}
        assert "EVENT_APPROVAL" not in ids


# ---------------------------------------------------------------------------
# Isolated database helpers
# ---------------------------------------------------------------------------

def _db_available() -> bool:
    """Check whether we can build an engine URL for hisho_test."""
    try:
        _make_isolated_engine()
        return True
    except Exception:
        return False


def _make_isolated_engine():
    """Create a fresh async engine pointing exclusively to hisho_test.

    Each async test must call this to get its own engine scoped to the current
    event loop, avoiding 'Event loop is closed' errors between tests.
    """
    raw_url = settings.database_url
    if not raw_url.startswith("postgresql+asyncpg://"):
        raw_url = raw_url.replace("postgresql://", "postgresql+asyncpg://")
    parsed = urllib.parse.urlparse(raw_url)
    test_url = parsed._replace(path="/hisho_test").geturl()
    return create_async_engine(test_url)


# Probe once at module load for the skipif decorator
_isolated_db_reachable = _db_available()


@pytest.mark.skipif(not _isolated_db_reachable, reason="Isolated test database (hisho_test) unavailable")
class TestIsolatedDatabaseSeeding:
    """Verify seeding and idempotency against the isolated test database (hisho_test).

    Each test method creates its own engine to avoid asyncpg / event-loop-closed
    errors that arise when multiple async tests share a single module-level engine.
    """

    @pytest.mark.asyncio
    async def test_seed_idempotency_on_isolated_db(self):
        """Seeding twice must produce exactly three records with zero duplicates."""
        engine = _make_isolated_engine()

        # 1. Ensure LetterType table exists in hisho_test and start clean
        async with engine.begin() as conn:
            await conn.run_sync(
                lambda c: Base.metadata.create_all(
                    c, tables=[LetterType.__table__, Official.__table__]
                )
            )
            await conn.execute(sa.text("DELETE FROM letter_type;"))
            await conn.execute(sa.text("DELETE FROM officials;"))

        session_factory = async_sessionmaker(engine, expire_on_commit=False)

        # 2. First seed run
        async with session_factory() as session:
            async with session.begin():
                seeded_first = await seed_sceptix_letter_types(session)
        assert len(seeded_first) == 3

        # Verify 3 records exist
        async with session_factory() as session:
            result = await session.execute(sa.select(LetterType))
            rows = result.scalars().all()
            assert len(rows) == 3
            assert {r.id for r in rows} == {"FINANCIAL_REQUEST", "EQUIPMENT_REQUEST", "VENUE_PERMISSION"}

            off_result = await session.execute(sa.select(Official))
            assert len(off_result.scalars().all()) == 0

        # 3. Second seed run (default overwrite=False)
        async with session_factory() as session:
            async with session.begin():
                seeded_second = await seed_sceptix_letter_types(session)
        assert len(seeded_second) == 3

        # Verify still exactly 3 records
        async with session_factory() as session:
            result = await session.execute(sa.select(LetterType))
            rows = result.scalars().all()
            assert len(rows) == 3

        # Cleanup
        async with engine.begin() as conn:
            await conn.execute(sa.text("DELETE FROM letter_type;"))
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_preserve_existing_config_on_no_overwrite(self):
        """Seeding with overwrite=False must not overwrite a customised description.

        Simulates an operator who has edited a LetterType description in production.
        Re-running the seed (default behavior) must not revert the customisation.
        """
        engine = _make_isolated_engine()

        async with engine.begin() as conn:
            await conn.run_sync(
                lambda c: Base.metadata.create_all(
                    c, tables=[LetterType.__table__, Official.__table__]
                )
            )
            await conn.execute(sa.text("DELETE FROM letter_type;"))

        session_factory = async_sessionmaker(engine, expire_on_commit=False)

        # First seed run
        async with session_factory() as session:
            async with session.begin():
                await seed_sceptix_letter_types(session)

        # Operator manually customises the FINANCIAL_REQUEST description
        custom_description = "CUSTOM: Operator-edited description that must survive re-seed."
        async with session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    sa.select(LetterType).where(LetterType.id == "FINANCIAL_REQUEST")
                )
                row = result.scalar_one()
                row.description = custom_description

        # Re-seed with overwrite=False (default)
        async with session_factory() as session:
            async with session.begin():
                await seed_sceptix_letter_types(session, overwrite=False)

        # Verify the customisation was preserved
        async with session_factory() as session:
            result = await session.execute(
                sa.select(LetterType).where(LetterType.id == "FINANCIAL_REQUEST")
            )
            row = result.scalar_one()
            assert row.description == custom_description, (
                f"Expected custom description to be preserved; got {row.description!r}"
            )

        # Cleanup
        async with engine.begin() as conn:
            await conn.execute(sa.text("DELETE FROM letter_type;"))
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_explicit_overwrite_reverts_to_seed_defaults(self):
        """Seeding with overwrite=True must replace a customised description with seed defaults."""
        engine = _make_isolated_engine()

        async with engine.begin() as conn:
            await conn.run_sync(
                lambda c: Base.metadata.create_all(
                    c, tables=[LetterType.__table__, Official.__table__]
                )
            )
            await conn.execute(sa.text("DELETE FROM letter_type;"))

        session_factory = async_sessionmaker(engine, expire_on_commit=False)

        # First seed run
        async with session_factory() as session:
            async with session.begin():
                await seed_sceptix_letter_types(session)

        seed_default_description = next(
            item["description"] for item in SCEPTIX_LETTER_TYPES
            if item["id"] == "FINANCIAL_REQUEST"
        )

        # Operator manually customises the description
        custom_description = "CUSTOM: This should be reverted when overwrite=True is used."
        async with session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    sa.select(LetterType).where(LetterType.id == "FINANCIAL_REQUEST")
                )
                row = result.scalar_one()
                row.description = custom_description

        # Re-seed with explicit overwrite=True
        async with session_factory() as session:
            async with session.begin():
                await seed_sceptix_letter_types(session, overwrite=True)

        # Verify the description was reverted to seed default
        async with session_factory() as session:
            result = await session.execute(
                sa.select(LetterType).where(LetterType.id == "FINANCIAL_REQUEST")
            )
            row = result.scalar_one()
            assert row.description == seed_default_description, (
                f"Expected seed default description after overwrite=True; got {row.description!r}"
            )
            assert row.description != custom_description

        # Cleanup
        async with engine.begin() as conn:
            await conn.execute(sa.text("DELETE FROM letter_type;"))
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_no_officials_seeded(self):
        """Seeding must never create rows in the officials table."""
        engine = _make_isolated_engine()

        async with engine.begin() as conn:
            await conn.run_sync(
                lambda c: Base.metadata.create_all(
                    c, tables=[LetterType.__table__, Official.__table__]
                )
            )
            await conn.execute(sa.text("DELETE FROM letter_type;"))
            await conn.execute(sa.text("DELETE FROM officials;"))

        session_factory = async_sessionmaker(engine, expire_on_commit=False)

        async with session_factory() as session:
            async with session.begin():
                await seed_sceptix_letter_types(session)

        async with session_factory() as session:
            result = await session.execute(sa.select(Official))
            officials = result.scalars().all()
            assert len(officials) == 0, (
                f"Expected no officials to be seeded; found {len(officials)} rows"
            )

        # Cleanup
        async with engine.begin() as conn:
            await conn.execute(sa.text("DELETE FROM letter_type;"))
        await engine.dispose()
