"""Tests for NodeTextFormatter: metadata preservation and composite searchable text."""

from llama_index.core.schema import TextNode
import pytest

from services.pipeline.pipeline_components.node_formatter import NodeTextFormatter


def test_metadata_preservation():
    """Verify that all required metadata fields survive formatting."""
    full_text = "To,\nThe Principal,\nSubject: Fund Release\n\nRespected Sir,\n..."
    raw_node = TextNode(
        id_="doc_123:tab_456",
        text="Temporary extraction text",
        metadata={
            "ref_no": "SCEPTIX/2026/01",
            "date": "12/03/2026",
            "subject": "Request for Fund Release",
            "letter_type": "FINANCIAL_REQUEST",
            "primary_domain": "FINANCE",
            "description": "Formal request seeking sanction of funds.",
            "table_info": {
                "summary": "Budget breakdown",
                "category": "FINANCIAL",
                "headers": ["Item", "Cost"],
            },
            "approval_chain": [
                {"step_number": 1, "designation_title": "Faculty Advisor", "incumbent": "Dr. A"}
            ],
            "signatures": [
                {"url": "https://example.com/sig.png", "name": "Jane Doe"}
            ],
            "source_doc_id": "doc_123:tab_456",
            "tab_title": "Winter Hackathon Letter",
            "full_markdown": full_text,
        },
    )

    formatter = NodeTextFormatter()
    result_nodes = formatter([raw_node])

    assert len(result_nodes) == 1
    node = result_nodes[0]

    # Verify all metadata fields survive
    meta = node.metadata
    assert meta["ref_no"] == "SCEPTIX/2026/01"
    assert meta["date"] == "12/03/2026"
    assert meta["subject"] == "Request for Fund Release"
    assert meta["letter_type"] == "FINANCIAL_REQUEST"
    assert meta["primary_domain"] == "FINANCE"
    assert meta["description"] == "Formal request seeking sanction of funds."
    assert meta["table_info"] == {
        "summary": "Budget breakdown",
        "category": "FINANCIAL",
        "headers": ["Item", "Cost"],
    }
    assert meta["approval_chain"] == [
        {"step_number": 1, "designation_title": "Faculty Advisor", "incumbent": "Dr. A"}
    ]
    assert meta["signatures"] == [
        {"url": "https://example.com/sig.png", "name": "Jane Doe"}
    ]
    assert meta["source_doc_id"] == "doc_123:tab_456"
    assert meta["tab_title"] == "Winter Hackathon Letter"
    assert meta["full_letter"] == full_text
    assert meta["full_markdown"] == full_text


def test_composite_searchable_text_composition():
    """Verify composite node.text encodes type, subject, purpose, and body."""
    raw_node = TextNode(
        id_="test_node_1",
        text="Intermediate description text",
        metadata={
            "letter_type": "VENUE_PERMISSION",
            "subject": "Permission for Prerna Hall",
            "description": "Formal request seeking venue allocation for technical workshop.",
            "full_markdown": "Respected Principal, we seek permission for Prerna Hall on 15 Oct.",
            "source_doc_id": "doc1:tab1",
            "tab_title": "Prerna Hall",
        },
    )

    formatter = NodeTextFormatter()
    result = formatter([raw_node])[0]

    text = result.text
    assert "Type: VENUE_PERMISSION" in text
    assert "Subject: Permission for Prerna Hall" in text
    assert "Purpose: Formal request seeking venue allocation for technical workshop." in text
    assert "Content:\nRespected Principal, we seek permission for Prerna Hall on 15 Oct." in text


def test_missing_optional_metadata_handling():
    """Verify missing optional fields omit sections gracefully without crashing or printing 'None'."""
    raw_node = TextNode(
        id_="test_node_missing",
        text="Some raw text",
        metadata={
            "subject": "Camera Request",
            "full_markdown": "Need camera equipment for streaming.",
            "source_doc_id": "doc2:tab1",
        },
    )

    formatter = NodeTextFormatter()
    result = formatter([raw_node])[0]

    text = result.text
    assert "Subject: Camera Request" in text
    assert "Content:\nNeed camera equipment for streaming." in text
    assert "Type:" not in text
    assert "Purpose:" not in text
    assert "None" not in text

    # Verify metadata fields default safely
    assert result.metadata["letter_type"] == ""
    assert result.metadata["description"] == ""
    assert result.metadata["table_info"] is None
    assert result.metadata["approval_chain"] == []


def test_failed_extraction_fallback():
    """Verify that when extraction completely fails, node.text safely falls back to available letter text."""
    raw_node = TextNode(
        id_="failed_node",
        text="Original letter text before extraction failed",
        metadata={
            "source_doc_id": "doc_fail:tab_1",
            "tab_title": "Failed Tab",
            "full_markdown": "Dear Sir,\nThis is the unextracted letter body.",
        },
    )

    formatter = NodeTextFormatter()
    result = formatter([raw_node])[0]

    # node.text should fall back to the body rather than being empty
    assert "Dear Sir,\nThis is the unextracted letter body." in result.text
    assert result.metadata["source_doc_id"] == "doc_fail:tab_1"
    assert result.metadata["full_letter"] == "Dear Sir,\nThis is the unextracted letter body."


def test_full_original_letter_preserved_without_truncation():
    """Verify that the full original letter is kept intact in metadata even when body in node.text is capped."""
    long_body = "Paragraph content repeat. " * 200  # ~5200 characters
    raw_node = TextNode(
        id_="long_node",
        text="desc",
        metadata={
            "letter_type": "FINANCIAL_REQUEST",
            "subject": "Comprehensive Financial Grant",
            "description": "Grant request.",
            "full_markdown": long_body,
            "source_doc_id": "doc_long:tab_1",
        },
    )

    formatter = NodeTextFormatter()
    result = formatter([raw_node])[0]

    # metadata["full_letter"] MUST have the full, untruncated content
    assert len(result.metadata["full_letter"]) == len(long_body)
    assert result.metadata["full_letter"] == long_body

    # node.text caps the body portion at 1500 chars to avoid vector dilution
    assert len(result.text) < len(long_body)
    assert result.text.startswith("Type: FINANCIAL_REQUEST")


def test_source_provenance_and_node_id_preserved():
    """Verify that node_id and provenance identifiers are preserved unchanged."""
    raw_node = TextNode(
        id_="stable_provenance_id_123",
        text="Some text",
        metadata={
            "source_doc_id": "google_doc_abc:tab_xyz",
            "tab_title": "Event Tab",
            "full_markdown": "Full text.",
        },
    )

    formatter = NodeTextFormatter()
    result = formatter([raw_node])[0]

    assert result.node_id == "stable_provenance_id_123"
    assert result.metadata["source_doc_id"] == "google_doc_abc:tab_xyz"
    assert result.metadata["tab_title"] == "Event Tab"


@pytest.mark.asyncio
async def test_acall_matches_sync_call():
    """Verify acall returns identical results to sync __call__."""
    raw_node = TextNode(
        id_="async_test",
        text="text",
        metadata={
            "letter_type": "VENUE_PERMISSION",
            "subject": "Hall Permission",
            "full_markdown": "Letter content",
            "source_doc_id": "doc:tab",
        },
    )

    formatter = NodeTextFormatter()
    sync_result = formatter([raw_node])
    async_result = await formatter.acall([raw_node])

    assert sync_result[0].text == async_result[0].text
    assert sync_result[0].metadata == async_result[0].metadata
