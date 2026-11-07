"""Idempotent seed script for Sceptix Club canonical letter types.

Establishes exactly three canonical operational categories derived from
the verified four-reference historical corpus:
1. FINANCIAL_REQUEST
2. EQUIPMENT_REQUEST
3. VENUE_PERMISSION

Rules:
- Classification is based strictly on Primary Requested Action, not event context.
- Prerna Hall is classified as VENUE_PERMISSION, never EVENT_APPROVAL.
- No Faker-generated officials or mock signatures are seeded.
- Category-level operational routing is kept separate from historical exemplar
  source mappings; historical variants (such as Faculty vs Club Coordinator)
  are documented rather than discarded.
- Does not invent unsupported approval stages (e.g. equipment request historical
  route has no HOD stage).
- Does not blend distinct administrative titles (e.g. venue permission recipient
  is Assistant Director, not Assistant Director / Principal).
- Safe by default: existing records are preserved unless explicit overwrite is requested.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import AsyncSessionLocal
from db.models import LetterType

logger = logging.getLogger(__name__)

# Canonical Sceptix letter type definitions for the operational catalogue.
# These represent baseline institutional configurations. Specific historical variants
# and document-level routing are preserved in services.constants.letter_types.
SCEPTIX_LETTER_TYPES: list[dict] = [
    {
        "id": "FINANCIAL_REQUEST",
        "name": "Financial / Budget Request",
        "description": (
            "Requests sanction, advance, or release of institutional or club funds for event budgets, "
            "hardware/asset procurement, competition prizes, participant logistics, or reimbursements. "
            "Characterized by itemized budget tables, total expenditure figures, and multi-tier administrative approval. "
            "Note: Historical exemplars show coordinator designation variants ('Faculty Coordinator' in Winter Hackathon "
            "and 'Club Coordinator' in HackToFuture4.0)."
        ),
        "approval_chain": [
            {
                "step": 1,
                "role": "Faculty Coordinator - The Sceptix Club",
                "action": "Verification & Endorsement",
            },
            {
                "step": 2,
                "role": "Head of Department - Computer Science & Engineering",
                "action": "Departmental Recommendation",
            },
            {
                "step": 3,
                "role": "Principal",
                "action": "Institutional Review & Forwarding",
            },
            {
                "step": 4,
                "role": "Director",
                "action": "Sanction & Final Financial Approval",
            },
        ],
        "template": (
            "From:\nThe Sceptix Club\nDepartment of Computer Science & Engineering\n"
            "St. Joseph Engineering College\n\n"
            "Date: <Current Date in House Style>\n\n"
            "To:\nThe Director\nSt. Joseph Engineering College\nVamanjoor, Mangaluru - 575028\n\n"
            "Through: The Principal\n"
            "Through: Head of Department – CSE\n"
            "Through: Faculty Coordinator – The Sceptix Club\n\n"
            "Subject: Request for sanction/release of funds for <Event / Purpose>\n\n"
            "Respected Sir,\n\n"
            "[Context & Justification]\n\n"
            "[Itemized Budget Table with Total Row]\n\n"
            "Thanking you,\n\nYours sincerely,\n[Signatures / Student Office Bearers]"
        ),
    },
    {
        "id": "EQUIPMENT_REQUEST",
        "name": "Equipment / Asset Requisition",
        "description": (
            "Requests temporary borrowing or requisition of physical equipment, audio-visual gear, "
            "cameras, tripods, streaming kits, laboratory hardware, or computing systems for club events or workshops. "
            "Specifies equipment schedule, custody assurance, and return commitment. "
            "Note: Historical exemplar routes directly to the facility in-charge (e.g. Central Computer Centre) "
            "through the Faculty Coordinator without an intermediate HOD stage."
        ),
        "approval_chain": [
            {
                "step": 1,
                "role": "Faculty Coordinator - The Sceptix Club",
                "action": "Verification & Endorsement",
            },
            {
                "step": 2,
                "role": "Central Computer Centre (CCC) / Facility In-Charge",
                "action": "Custodial Approval & Asset Allocation",
            },
        ],
        "template": (
            "From:\nThe Sceptix Club\nDepartment of Computer Science & Engineering\n"
            "St. Joseph Engineering College\n\n"
            "Date: <Current Date in House Style>\n\n"
            "To:\nThe In-Charge, Central Computer Centre (CCC) / Facility In-Charge\n"
            "St. Joseph Engineering College\n\n"
            "Through: Faculty Coordinator – The Sceptix Club\n\n"
            "Subject: Request to borrow/requisition equipment for <Workshop / Event>\n\n"
            "Respected Madam/Sir,\n\n"
            "[Details of required equipment, custody undertaking, and duration]\n\n"
            "Thanking you,\n\nYours sincerely,\n[Signatures / Student Office Bearers]"
        ),
    },
    {
        "id": "VENUE_PERMISSION",
        "name": "Venue / Hall Booking Permission",
        "description": (
            "Requests permission to book, access, or utilize campus facilities, institutional halls (such as Prerna Hall), "
            "auditoriums, seminar halls, classrooms, or computer labs for a designated date and duration, "
            "regardless of the event context. "
            "Note: Historical exemplar is addressed to the Assistant Director, routed through the Faculty Coordinator "
            "and Head of Department - CSE."
        ),
        "approval_chain": [
            {
                "step": 1,
                "role": "Faculty Coordinator - The Sceptix Club",
                "action": "Verification & Endorsement",
            },
            {
                "step": 2,
                "role": "Head of Department - Computer Science & Engineering",
                "action": "Departmental Recommendation",
            },
            {
                "step": 3,
                "role": "Assistant Director",
                "action": "Facility Allocation & Final Approval",
            },
        ],
        "template": (
            "From:\nThe Sceptix Club\nDepartment of Computer Science & Engineering\n"
            "St. Joseph Engineering College\n\n"
            "Date: <Current Date in House Style>\n\n"
            "To:\nThe Assistant Director\nSt. Joseph Engineering College\nVamanjoor, Mangaluru - 575028\n\n"
            "Through: Faculty Coordinator – The Sceptix Club\n"
            "Through: Head of Department – CSE\n\n"
            "Subject: Permission to utilize <Facility Name> for <Event Name>\n\n"
            "Respected Father/Sir,\n\n"
            "[Event description, schedule, venue requirements, and student attendance estimation]\n\n"
            "Thanking you,\n\nYours sincerely,\n[Signatures / Student Office Bearers]"
        ),
    },
]


async def seed_sceptix_letter_types(
    session: AsyncSession,
    overwrite: bool = False,
) -> list[str]:
    """Seed or update the canonical Sceptix letter types safely and idempotently.

    Default behavior (overwrite=False):
    - If a LetterType record with the canonical id exists, its existing attributes
      (name, description, approval_chain, template) are preserved intact.
    - Missing records are inserted.

    Intentional update behavior (overwrite=True):
    - Existing records are updated with the seed defaults.

    Args:
        session: Active SQLAlchemy AsyncSession.
        overwrite: If True, overwrite existing records. Default is False.

    Returns:
        List of letter_type IDs that were seeded or preserved.
    """
    seeded_ids: list[str] = []

    for item in SCEPTIX_LETTER_TYPES:
        stmt = select(LetterType).where(LetterType.id == item["id"])
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            if overwrite:
                existing.name = item["name"]
                existing.description = item["description"]
                existing.approval_chain = item["approval_chain"]
                existing.template = item["template"]
                logger.info("Updated existing LetterType '%s' (overwrite=True)", item["id"])
            else:
                logger.info("Preserving existing LetterType '%s' (overwrite=False)", item["id"])
        else:
            session.add(
                LetterType(
                    id=item["id"],
                    name=item["name"],
                    description=item["description"],
                    approval_chain=item["approval_chain"],
                    template=item["template"],
                )
            )
            logger.info("Inserted new LetterType '%s'", item["id"])
        seeded_ids.append(item["id"])

    await session.flush()
    return seeded_ids


async def run_seed(confirm: bool = False, overwrite: bool = False) -> list[str]:
    """Execute the seed operation against the configured database.

    Args:
        confirm: Must be True to proceed with database execution.
        overwrite: If True, updates existing LetterType rows with seed defaults.
                   Default is False (preserves existing configurations).
    """
    if not confirm:
        raise RuntimeError(
            "Execution against database requires explicit confirmation. "
            "Pass confirm=True or use --confirm on CLI."
        )

    async with AsyncSessionLocal() as session:
        async with session.begin():
            seeded_ids = await seed_sceptix_letter_types(session, overwrite=overwrite)
            logger.info(
                "Successfully processed %d Sceptix letter types (overwrite=%s): %s",
                len(seeded_ids),
                overwrite,
                seeded_ids,
            )
            return seeded_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Sceptix Club canonical letter types.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Explicit confirmation to execute against the configured database.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly overwrite existing letter type definitions with seed defaults.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(run_seed(confirm=args.confirm, overwrite=args.overwrite))
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
