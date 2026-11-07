"""Unit tests for Sceptix Club three-category taxonomy and constants."""

import pytest
from services.constants.letter_types import (
    CANONICAL_SCEPTIX_CATEGORIES,
    LETTER_TYPE_CATEGORIES,
    HISTORICAL_REFERENCE_DOCUMENTS,
    HISTORICAL_REFERENCE_MAPPINGS,
    HistoricalReferenceDoc,
    LetterTypeCategory,
    derive_letter_display_order,
    format_letter_types_block,
)


class TestSceptixTaxonomyConstants:
    """Validate canonical category definitions and classification rules."""

    def test_canonical_categories_count_and_keys(self):
        """Initial reference corpus must have exactly three canonical categories."""
        expected_keys = {"FINANCIAL_REQUEST", "EQUIPMENT_REQUEST", "VENUE_PERMISSION"}
        actual_keys = {cat.value for cat in CANONICAL_SCEPTIX_CATEGORIES}
        assert actual_keys == expected_keys
        assert len(CANONICAL_SCEPTIX_CATEGORIES) == 3

    def test_event_approval_excluded(self):
        """EVENT_APPROVAL must not be in the canonical or operational catalogue."""
        active_values = {cat.value for cat in LETTER_TYPE_CATEGORIES}
        assert "EVENT_APPROVAL" not in active_values
        assert "ACADEMIC_DUTY_LEAVE" not in active_values
        assert "OFFICIAL_CIRCULAR" not in active_values
        assert "INVITATION" not in active_values

    def test_categories_have_meaningful_descriptions(self):
        """Every canonical category must provide a descriptive instruction."""
        for cat in CANONICAL_SCEPTIX_CATEGORIES:
            assert isinstance(cat.value, str) and len(cat.value) > 0
            assert isinstance(cat.description, str) and len(cat.description) > 20
            # Str representation format
            assert str(cat) == f"{cat.value} ({cat.description})"

    def test_format_letter_types_block_renders_all_three(self):
        """Prompt formatter must include bullet points for all canonical categories."""
        block = format_letter_types_block()
        assert "- FINANCIAL_REQUEST" in block
        assert "- EQUIPMENT_REQUEST" in block
        assert "- VENUE_PERMISSION" in block
        assert "EVENT_APPROVAL" not in block


class TestHistoricalReferenceMappings:
    """Validate that the four historical reference documents map by Primary Action."""

    def test_four_reference_documents_present(self):
        """All four reference documents must be present in HISTORICAL_REFERENCE_DOCUMENTS."""
        expected_titles = {
            "Winter Hackathon",
            "Transfer of Funds to HackToFuture4.0",
            "Camera Setup for OSM",
            "Prerna Hall for Tech Trivia",
        }
        actual_titles = {doc.document_title for doc in HISTORICAL_REFERENCE_DOCUMENTS}
        assert actual_titles == expected_titles
        assert len(HISTORICAL_REFERENCE_DOCUMENTS) == 4

    def test_winter_hackathon_is_financial(self):
        """Winter Hackathon is classified as FINANCIAL_REQUEST."""
        mapping = HISTORICAL_REFERENCE_MAPPINGS["Winter Hackathon"]
        assert mapping["canonical_category"] == "FINANCIAL_REQUEST"
        assert "funds" in mapping["primary_action"].lower()

    def test_hacktofuture_is_financial(self):
        """HackToFuture4.0 alias is classified as FINANCIAL_REQUEST."""
        mapping = HISTORICAL_REFERENCE_MAPPINGS["HackToFuture4.0"]
        assert mapping["canonical_category"] == "FINANCIAL_REQUEST"
        assert "funds" in mapping["primary_action"].lower()

    def test_camera_equipment_is_equipment(self):
        """Camera Equipment for OSM Workshop alias is classified as EQUIPMENT_REQUEST."""
        mapping = HISTORICAL_REFERENCE_MAPPINGS["Camera Equipment for OSM Workshop"]
        assert mapping["canonical_category"] == "EQUIPMENT_REQUEST"
        assert "equipment" in mapping["primary_action"].lower()

    def test_prerna_hall_is_venue_permission_not_event_approval(self):
        """Prerna Hall for Tech Trivia must be VENUE_PERMISSION based on primary action."""
        mapping = HISTORICAL_REFERENCE_MAPPINGS["Prerna Hall for Tech Trivia"]
        assert mapping["canonical_category"] == "VENUE_PERMISSION"
        assert mapping["canonical_category"] != "EVENT_APPROVAL"
        # classification_notes must distinguish from EVENT_APPROVAL
        notes = mapping["classification_notes"]
        assert "VENUE_PERMISSION" in notes
        assert "EVENT_APPROVAL" in notes


# ---------------------------------------------------------------------------
# 10 required new tests
# ---------------------------------------------------------------------------


class TestHistoricalDocumentCategories:
    """Test 1 – All four documents map to their correct canonical categories."""

    def test_all_four_documents_canonical_categories(self):
        """Every reference document must resolve to one of the three canonical categories."""
        expected = {
            "Winter Hackathon": "FINANCIAL_REQUEST",
            "Transfer of Funds to HackToFuture4.0": "FINANCIAL_REQUEST",
            "Camera Setup for OSM": "EQUIPMENT_REQUEST",
            "Prerna Hall for Tech Trivia": "VENUE_PERMISSION",
        }
        for doc in HISTORICAL_REFERENCE_DOCUMENTS:
            assert doc.canonical_category == expected[doc.document_title], (
                f"{doc.document_title} mapped to {doc.canonical_category!r}, "
                f"expected {expected[doc.document_title]!r}"
            )


class TestCoordinatorDesignationVariants:
    """Tests 2 & 3 – Coordinator designation variants are preserved, not standardised."""

    def test_winter_hackathon_preserves_faculty_coordinator(self):
        """Test 2: Winter Hackathon must carry 'Faculty Coordinator' designation."""
        doc = next(
            d for d in HISTORICAL_REFERENCE_DOCUMENTS
            if d.document_title == "Winter Hackathon"
        )
        assert "Faculty Coordinator" in doc.coordinator_designation
        assert "Club Coordinator" not in doc.coordinator_designation

    def test_hacktofuture_preserves_club_coordinator(self):
        """Test 3: Transfer of Funds to HackToFuture4.0 must carry 'Club Coordinator' designation."""
        doc = next(
            d for d in HISTORICAL_REFERENCE_DOCUMENTS
            if d.document_title == "Transfer of Funds to HackToFuture4.0"
        )
        assert "Club Coordinator" in doc.coordinator_designation
        assert "Faculty Coordinator" not in doc.coordinator_designation


class TestEquipmentRequestWorkflowOrder:
    """Test 4 – Equipment request workflow_order contains no HOD stage."""

    def test_equipment_workflow_order_has_no_hod(self):
        """Test 4: Camera Setup for OSM workflow_order must not include an HOD stage."""
        doc = next(
            d for d in HISTORICAL_REFERENCE_DOCUMENTS
            if d.document_title == "Camera Setup for OSM"
        )
        for stage in doc.workflow_order:
            assert "HOD" not in stage.upper(), (
                f"Unexpected HOD reference in equipment workflow stage: {stage!r}"
            )
            assert "Head of Department" not in stage, (
                f"Unexpected HOD reference in equipment workflow stage: {stage!r}"
            )


class TestVenuePermissionRecipient:
    """Test 5 – Venue permission addressed to Assistant Director only."""

    def test_prerna_hall_recipient_is_assistant_director_not_blended(self):
        """Test 5: Prerna Hall recipient must be 'The Assistant Director', not a blended designation."""
        doc = next(
            d for d in HISTORICAL_REFERENCE_DOCUMENTS
            if d.document_title == "Prerna Hall for Tech Trivia"
        )
        assert doc.recipient == "The Assistant Director", (
            f"Expected 'The Assistant Director', got {doc.recipient!r}"
        )
        # Must not silently blend 'Assistant Director / Principal'
        assert "Principal" not in doc.recipient
        assert "/" not in doc.recipient


class TestWorkflowOrderBottomUp:
    """Test 6 – workflow_order is bottom-up: first endorser at index 0, final decider last."""

    def test_financial_workflow_order_is_bottom_up(self):
        """Test 6: Faculty/Club Coordinator must be first; Director must be last."""
        for doc in HISTORICAL_REFERENCE_DOCUMENTS:
            if doc.canonical_category != "FINANCIAL_REQUEST":
                continue
            assert len(doc.workflow_order) >= 2, (
                f"{doc.document_title} workflow_order must have at least 2 stages"
            )
            first = doc.workflow_order[0]
            last = doc.workflow_order[-1]
            assert "Coordinator" in first, (
                f"{doc.document_title}: Expected coordinator first, got {first!r}"
            )
            assert "Director" in last, (
                f"{doc.document_title}: Expected Director last, got {last!r}"
            )

    def test_prerna_hall_workflow_order_is_bottom_up(self):
        """Prerna Hall workflow order must be Coordinator -> HOD -> Assistant Director."""
        doc = next(
            d for d in HISTORICAL_REFERENCE_DOCUMENTS
            if d.document_title == "Prerna Hall for Tech Trivia"
        )
        assert doc.workflow_order == [
            "Ms Pruthvi M R – Faculty Coordinator, The Sceptix Club",
            "Head of Department – CSE",
            "The Assistant Director",
        ]


class TestHistoricalSourceLetterDisplayOrders:
    """Test 7 – Source-specific letter display orders for all four historical documents.

    Validates that letter display order is preserved exactly as physically formatted
    in each historical reference document, without assuming a universal inversion
    or derivation from workflow order.
    """

    def test_winter_hackathon_display_order(self):
        """Winter Hackathon: To Director; Through Principal -> HOD -> Faculty Coordinator."""
        doc = next(
            d for d in HISTORICAL_REFERENCE_DOCUMENTS
            if d.document_title == "Winter Hackathon"
        )
        display = doc.letter_display_order
        assert display["To"] == "The Director"
        assert display["Through"] == [
            "The Principal",
            "Head of Department – CSE",
            "Ms Pruthvi M R – Faculty Coordinator, The Sceptix Club",
        ]

    def test_hacktofuture_display_order(self):
        """HackToFuture4.0: To Director; Through Principal -> HOD -> Club Coordinator."""
        doc = next(
            d for d in HISTORICAL_REFERENCE_DOCUMENTS
            if d.document_title == "Transfer of Funds to HackToFuture4.0"
        )
        display = doc.letter_display_order
        assert display["To"] == "The Director"
        assert display["Through"] == [
            "The Principal",
            "Head of Department – CSE",
            "Ms Pruthvi M R – Club Coordinator, The Sceptix Club",
        ]

    def test_camera_setup_osm_display_order(self):
        """Camera Setup for OSM: To CCC In-Charge; Through Faculty Coordinator."""
        doc = next(
            d for d in HISTORICAL_REFERENCE_DOCUMENTS
            if d.document_title == "Camera Setup for OSM"
        )
        display = doc.letter_display_order
        assert display["To"] == "Ms Ida Jacintha Cordeiro – CCC"
        assert display["Through"] == [
            "Ms Pruthvi M R – Faculty Coordinator, The Sceptix Club",
        ]

    def test_prerna_hall_display_order_preserves_source_order(self):
        """Prerna Hall: To Assistant Director; Through Faculty Coordinator -> HOD.

        The physical document displays the 'Through' block starting with
        Faculty Coordinator followed by HOD, unlike the financial letters
        which display top-down (Principal first). This source-specific
        difference must be preserved.
        """
        doc = next(
            d for d in HISTORICAL_REFERENCE_DOCUMENTS
            if d.document_title == "Prerna Hall for Tech Trivia"
        )
        display = doc.letter_display_order
        assert display["To"] == "The Assistant Director"
        assert display["Through"] == [
            "Ms Pruthvi M R – Faculty Coordinator, The Sceptix Club",
            "Head of Department – CSE",
        ]

    def test_display_order_not_universally_inverted_from_workflow(self):
        """Display order must not be universally assumed to be reversed(workflow_order[:-1]).

        For Prerna Hall, display Through is [Coordinator, HOD], which equals
        workflow_order[:-1], NOT reversed(workflow_order[:-1]).
        """
        doc = next(
            d for d in HISTORICAL_REFERENCE_DOCUMENTS
            if d.document_title == "Prerna Hall for Tech Trivia"
        )
        display_through = doc.letter_display_order["Through"]
        reversed_workflow = list(reversed(doc.workflow_order[:-1]))
        assert display_through != reversed_workflow, (
            "Prerna Hall display order must NOT match reversed workflow order"
        )
        assert display_through == doc.workflow_order[:-1], (
            "Prerna Hall display Through must match its actual source document order [Coordinator, HOD]"
        )


class TestDeriveLetterDisplayOrderUtility:
    """Test 8 – derive_letter_display_order() utility returns correct structure."""

    def test_derive_letter_display_order_simple_chain(self):
        """Test 8: utility correctly splits bottom-up list into To/Through display dict."""
        chain = ["Faculty Coordinator", "HOD", "Principal", "Director"]
        result = derive_letter_display_order(chain)
        assert result["To"] == "Director"
        assert result["Through"] == ["Faculty Coordinator", "HOD", "Principal"]

    def test_derive_letter_display_order_two_step(self):
        """Test 8b: two-step chain produces single-step Through."""
        chain = ["Faculty Coordinator", "CCC In-Charge"]
        result = derive_letter_display_order(chain)
        assert result["To"] == "CCC In-Charge"
        assert result["Through"] == ["Faculty Coordinator"]

    def test_derive_letter_display_order_empty(self):
        """Test 8c: empty chain returns empty To and Through."""
        result = derive_letter_display_order([])
        assert result["To"] == ""
        assert result["Through"] == []


class TestFormatLetterTypesBlockNoEventApproval:
    """Test 9 – format_letter_types_block() does not include EVENT_APPROVAL."""

    def test_format_block_excludes_unsupported_categories(self):
        """Test 9: formatted block must contain only the three canonical categories."""
        block = format_letter_types_block()
        assert "EVENT_APPROVAL" not in block
        assert "ACADEMIC_DUTY_LEAVE" not in block
        assert "OFFICIAL_CIRCULAR" not in block
        assert "INVITATION" not in block
        assert "- FINANCIAL_REQUEST" in block
        assert "- EQUIPMENT_REQUEST" in block
        assert "- VENUE_PERMISSION" in block


class TestHistoricalReferenceMappingsAliases:
    """Test 10 – HISTORICAL_REFERENCE_MAPPINGS aliases work for both short and canonical names."""

    def test_hacktofuture_alias_accessible(self):
        """Test 10a: 'HackToFuture4.0' alias must resolve to the same record as the canonical title."""
        alias = HISTORICAL_REFERENCE_MAPPINGS["HackToFuture4.0"]
        canonical = HISTORICAL_REFERENCE_MAPPINGS["Transfer of Funds to HackToFuture4.0"]
        assert alias is canonical, (
            "Alias 'HackToFuture4.0' must point to the same dict object as the canonical title"
        )
        assert alias["canonical_category"] == "FINANCIAL_REQUEST"

    def test_camera_equipment_alias_accessible(self):
        """Test 10b: 'Camera Equipment for OSM Workshop' alias must resolve correctly."""
        alias = HISTORICAL_REFERENCE_MAPPINGS["Camera Equipment for OSM Workshop"]
        canonical = HISTORICAL_REFERENCE_MAPPINGS["Camera Setup for OSM"]
        assert alias is canonical, (
            "Alias 'Camera Equipment for OSM Workshop' must point to the same dict object "
            "as the canonical title 'Camera Setup for OSM'"
        )
        assert alias["canonical_category"] == "EQUIPMENT_REQUEST"

    def test_all_primary_titles_accessible_directly(self):
        """Test 10c: All four canonical document titles must be directly accessible."""
        primary_titles = [
            "Winter Hackathon",
            "Transfer of Funds to HackToFuture4.0",
            "Camera Setup for OSM",
            "Prerna Hall for Tech Trivia",
        ]
        for title in primary_titles:
            assert title in HISTORICAL_REFERENCE_MAPPINGS, (
                f"Primary title {title!r} not found in HISTORICAL_REFERENCE_MAPPINGS"
            )

    def test_historical_reference_docs_are_dataclass_instances(self):
        """Test 10d: HISTORICAL_REFERENCE_DOCUMENTS entries must be HistoricalReferenceDoc instances."""
        for doc in HISTORICAL_REFERENCE_DOCUMENTS:
            assert isinstance(doc, HistoricalReferenceDoc), (
                f"Expected HistoricalReferenceDoc, got {type(doc)!r}"
            )
