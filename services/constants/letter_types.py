"""Letter type categories for the Sceptix Club reference corpus and extraction.

Establishes exactly three canonical categories based on the Primary Requested Action:
1. FINANCIAL_REQUEST (sanction, advance, budget release, procurement, participation expenses, prizes)
2. EQUIPMENT_REQUEST (temporary borrowing or requisition of AV gear, cameras, systems)
3. VENUE_PERMISSION (permission to book or use institutional halls, classrooms, labs)

Classification is strictly based on the primary requested administrative action,
never the event context (e.g. Prerna Hall for Tech Trivia is VENUE_PERMISSION,
not EVENT_APPROVAL).

Maintains exact historical source routing and display order for the four verified reference documents,
distinguishing bottom-up workflow approval order from source-specific letter display order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class LetterTypeCategory:
    value: str
    description: str

    def __str__(self) -> str:
        return f"{self.value} ({self.description})"


# Exactly three canonical operational categories for the Sceptix Club reference corpus
CANONICAL_SCEPTIX_CATEGORIES: list[LetterTypeCategory] = [
    LetterTypeCategory(
        value="FINANCIAL_REQUEST",
        description=(
            "Sanction, advance, allocation, or release of club/institutional funds for event budgets, "
            "hardware or asset procurement, competition prizes, participant logistics, or reimbursements."
        ),
    ),
    LetterTypeCategory(
        value="EQUIPMENT_REQUEST",
        description=(
            "Temporary borrowing or requisition of physical assets, audio-visual equipment, cameras, "
            "systems, display hardware, or laboratory kits."
        ),
    ),
    LetterTypeCategory(
        value="VENUE_PERMISSION",
        description=(
            "Permission to book, access, or utilize campus facilities, institutional halls (such as Prerna Hall), "
            "auditoriums, seminar halls, classrooms, or computer labs, regardless of event context."
        ),
    ),
]

# The single source of truth for operational categories presented to models and routing
LETTER_TYPE_CATEGORIES: list[LetterTypeCategory] = CANONICAL_SCEPTIX_CATEGORIES


@dataclass(frozen=True)
class HistoricalReferenceDoc:
    """Exact source-specific metadata extracted from a historical reference letter."""

    document_title: str
    canonical_category: str
    primary_action: str
    recipient: str
    coordinator_designation: str
    # Sequential bottom-up administrative routing (from initial endorser up to final decider)
    workflow_order: list[str] = field(default_factory=list)
    # Visual letter display structure as physically rendered on the document
    letter_display_order: dict[str, Any] = field(default_factory=dict)
    classification_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_title": self.document_title,
            "canonical_category": self.canonical_category,
            "primary_action": self.primary_action,
            "recipient": self.recipient,
            "coordinator_designation": self.coordinator_designation,
            "workflow_order": list(self.workflow_order),
            "letter_display_order": dict(self.letter_display_order),
            "historical_routing": " -> ".join(self.workflow_order),
            "classification_notes": self.classification_notes,
        }


# The four historical reference documents with their verified source-specific routing
HISTORICAL_REFERENCE_DOCUMENTS: list[HistoricalReferenceDoc] = [
    HistoricalReferenceDoc(
        document_title="Winter Hackathon",
        canonical_category="FINANCIAL_REQUEST",
        primary_action="Sanction and release of funds for competition prizes, equipment procurement, and event arrangements",
        recipient="The Director",
        coordinator_designation="Faculty Coordinator, The Sceptix Club",
        workflow_order=[
            "Ms Pruthvi M R – Faculty Coordinator, The Sceptix Club",
            "Head of Department – CSE",
            "The Principal",
            "The Director",
        ],
        letter_display_order={
            "To": "The Director",
            "Through": [
                "The Principal",
                "Head of Department – CSE",
                "Ms Pruthvi M R – Faculty Coordinator, The Sceptix Club",
            ],
        },
        classification_notes="Financial request with Faculty Coordinator endorsement.",
    ),
    HistoricalReferenceDoc(
        document_title="Transfer of Funds to HackToFuture4.0",
        canonical_category="FINANCIAL_REQUEST",
        primary_action="Sanction and release of funds for participation logistics and hackathon expenses",
        recipient="The Director",
        coordinator_designation="Club Coordinator, The Sceptix Club",
        workflow_order=[
            "Ms Pruthvi M R – Club Coordinator, The Sceptix Club",
            "Head of Department – CSE",
            "The Principal",
            "The Director",
        ],
        letter_display_order={
            "To": "The Director",
            "Through": [
                "The Principal",
                "Head of Department – CSE",
                "Ms Pruthvi M R – Club Coordinator, The Sceptix Club",
            ],
        },
        classification_notes="Coordinator designation in archive is Club Coordinator, distinguishable from Faculty Coordinator.",
    ),
    HistoricalReferenceDoc(
        document_title="Camera Setup for OSM",
        canonical_category="EQUIPMENT_REQUEST",
        primary_action="Temporary borrowing of audio-visual and streaming camera equipment",
        recipient="Ms Ida Jacintha Cordeiro – CCC",
        coordinator_designation="Faculty Coordinator, The Sceptix Club",
        workflow_order=[
            "Ms Pruthvi M R – Faculty Coordinator, The Sceptix Club",
            "Ms Ida Jacintha Cordeiro – CCC",
        ],
        letter_display_order={
            "To": "Ms Ida Jacintha Cordeiro – CCC",
            "Through": [
                "Ms Pruthvi M R – Faculty Coordinator, The Sceptix Club",
            ],
        },
        classification_notes="Addressed directly to CCC facility in-charge; does not include an HOD routing stage.",
    ),
    HistoricalReferenceDoc(
        document_title="Prerna Hall for Tech Trivia",
        canonical_category="VENUE_PERMISSION",
        primary_action="Permission to utilize Prerna Hall for hosting a technical competition",
        recipient="The Assistant Director",
        coordinator_designation="Faculty Coordinator, The Sceptix Club",
        workflow_order=[
            "Ms Pruthvi M R – Faculty Coordinator, The Sceptix Club",
            "Head of Department – CSE",
            "The Assistant Director",
        ],
        letter_display_order={
            "To": "The Assistant Director",
            "Through": [
                "Ms Pruthvi M R – Faculty Coordinator, The Sceptix Club",
                "Head of Department – CSE",
            ],
        },
        classification_notes="Classified strictly as VENUE_PERMISSION based on the requested administrative action, not EVENT_APPROVAL. Addressed to Assistant Director, not Principal.",
    ),
]


# Canonical lookup dictionary supporting standard document titles and aliases
HISTORICAL_REFERENCE_MAPPINGS: dict[str, dict[str, Any]] = {}
for _doc in HISTORICAL_REFERENCE_DOCUMENTS:
    _d = _doc.to_dict()
    HISTORICAL_REFERENCE_MAPPINGS[_doc.document_title] = _d
    # Aliases
    if _doc.document_title == "Transfer of Funds to HackToFuture4.0":
        HISTORICAL_REFERENCE_MAPPINGS["HackToFuture4.0"] = _d
    elif _doc.document_title == "Camera Setup for OSM":
        HISTORICAL_REFERENCE_MAPPINGS["Camera Equipment for OSM Workshop"] = _d


def derive_letter_display_order(workflow_order: list[str]) -> dict[str, Any]:
    """Derive visual letter display structure from bottom-up workflow routing.

    Standard convention helper:
    - The final entry in the workflow order is the recipient ('To').
    - Intermediate endorsers constitute the 'Through' routing block.

    Note: Historical documents retain their source-specific letter display
    ordering, which may follow either bottom-up or top-down conventions
    depending on the document.
    """
    if not workflow_order:
        return {"To": "", "Through": []}

    recipient = workflow_order[-1]
    through_steps = workflow_order[:-1]
    return {"To": recipient, "Through": through_steps}


def format_letter_types_block(categories: Sequence[LetterTypeCategory] | None = None) -> str:
    """Render the bullet list of letter types for extraction and agent prompts."""
    target = categories if categories is not None else LETTER_TYPE_CATEGORIES
    return "\n".join(f"- {category}" for category in target)


# Set of valid canonical category identifiers
CANONICAL_CATEGORY_IDS: frozenset[str] = frozenset(
    cat.value for cat in CANONICAL_SCEPTIX_CATEGORIES
)

# Known semantic aliases mapped to canonical categories
_CATEGORY_ALIASES: dict[str, str] = {
    # Financial
    "FINANCIAL": "FINANCIAL_REQUEST",
    "FINANCE": "FINANCIAL_REQUEST",
    "BUDGET_REQUEST": "FINANCIAL_REQUEST",
    "BUDGET": "FINANCIAL_REQUEST",
    "FUNDS_REQUEST": "FINANCIAL_REQUEST",
    "FUND_RELEASE": "FINANCIAL_REQUEST",
    "FINANCIAL_APPROVAL": "FINANCIAL_REQUEST",
    # Equipment
    "EQUIPMENT": "EQUIPMENT_REQUEST",
    "ASSET_REQUISITION": "EQUIPMENT_REQUEST",
    "HARDWARE_REQUEST": "EQUIPMENT_REQUEST",
    "BORROW_EQUIPMENT": "EQUIPMENT_REQUEST",
    "EQUIPMENT_BORROWING": "EQUIPMENT_REQUEST",
    # Venue
    "VENUE": "VENUE_PERMISSION",
    "VENUE_BOOKING": "VENUE_PERMISSION",
    "HALL_BOOKING": "VENUE_PERMISSION",
    "CLASSROOM_PERMISSION": "VENUE_PERMISSION",
    "VENUE_ALLOCATION": "VENUE_PERMISSION",
}


def normalize_canonical_category(
    raw_category: str | None,
    document_title: str | None = None,
) -> str:
    """Normalize and validate an extraction category against the canonical Sceptix taxonomy.

    Enforces exactly three canonical categories:
    - FINANCIAL_REQUEST
    - EQUIPMENT_REQUEST
    - VENUE_PERMISSION

    Rules:
    1. Historical reference documents map deterministically by title:
       - 'Winter Hackathon' -> FINANCIAL_REQUEST
       - 'Transfer of Funds to HackToFuture4.0' (and aliases) -> FINANCIAL_REQUEST
       - 'Camera Setup for OSM' (and aliases) -> EQUIPMENT_REQUEST
       - 'Prerna Hall for Tech Trivia' -> VENUE_PERMISSION (never EVENT_APPROVAL)
    2. Classification follows the Primary Requested Action, not event context.
    3. Normalizes common formatting variations (casing, whitespace, hyphens).
    4. Unsupported categories (e.g. EVENT_APPROVAL, ACADEMIC_DUTY_LEAVE, INVITATION)
       are rejected explicitly with ValueError. No silent fallback to arbitrary categories.

    Args:
        raw_category: Category string produced by extraction or user selection.
        document_title: Optional document/tab title used to disambiguate historical references.

    Returns:
        Canonical category identifier.

    Raises:
        ValueError: If the category is invalid or unsupported.
    """
    cleaned_title = (document_title or "").strip()

    # 1. Deterministic historical reference document mapping
    if cleaned_title:
        # Exact title or alias lookup
        if cleaned_title in HISTORICAL_REFERENCE_MAPPINGS:
            return HISTORICAL_REFERENCE_MAPPINGS[cleaned_title]["canonical_category"]

        # Substring title checks for verified reference exemplars
        title_lower = cleaned_title.lower()
        if "prerna hall" in title_lower or "tech trivia" in title_lower:
            return "VENUE_PERMISSION"
        if "winter hackathon" in title_lower:
            return "FINANCIAL_REQUEST"
        if "hacktofuture" in title_lower:
            return "FINANCIAL_REQUEST"
        if "camera" in title_lower and "osm" in title_lower:
            return "EQUIPMENT_REQUEST"

    # 2. Validate and normalize raw_category
    if not raw_category or not raw_category.strip():
        raise ValueError(
            "Cannot normalize category: both category and document title are missing or unresolvable."
        )

    normalized = raw_category.strip().upper().replace(" ", "_").replace("-", "_")

    # Direct canonical match
    if normalized in CANONICAL_CATEGORY_IDS:
        return normalized

    # Known alias match
    if normalized in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[normalized]

    # Explicit rejection of EVENT_APPROVAL
    if normalized == "EVENT_APPROVAL":
        raise ValueError(
            "Unsupported category 'EVENT_APPROVAL': Institutional correspondence must be "
            "classified by the Primary Requested Action (e.g. VENUE_PERMISSION, FINANCIAL_REQUEST), "
            "not the event context."
        )

    # Any other unsupported category
    supported = ", ".join(sorted(CANONICAL_CATEGORY_IDS))
    raise ValueError(
        f"Invalid or unsupported letter category '{raw_category}'. "
        f"Exactly three canonical categories are supported: {supported}."
    )
