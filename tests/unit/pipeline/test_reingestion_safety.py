"""Isolated tests demonstrating re-ingestion behaviors and edge cases.

These tests run purely in-memory using mocks without connecting to or mutating
any live database.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from llama_index.core.schema import TextNode
import pytest

from services.pipeline.pipeline_components.gdoc_parser import GoogleDocJsonParser
from services.pipeline.pipeline_components.node_formatter import NodeTextFormatter
from services.pipeline.pipeline_components.legacy_sync import LegacyTableSync


def test_first_ingestion_node_construction():
    """Test 1: First ingestion properly assigns source_doc_id to node.id_ and preserves metadata."""
    node = TextNode(
        text="Dear Principal,\nWe request venue permission.",
        metadata={
            "source_doc_id": "doc_alpha:tab_01",
            "tab_title": "Venue Tab",
            "full_markdown": "Dear Principal,\nWe request venue permission.",
            "letter_type": "VENUE_PERMISSION",
            "subject": "Venue Permission",
            "description": "Formal request seeking venue allocation.",
        },
    )
    formatter = NodeTextFormatter()
    res = formatter([node])[0]

    assert res.metadata["source_doc_id"] == "doc_alpha:tab_01"
    assert "Type: VENUE_PERMISSION" in res.text
    assert "Subject: Venue Permission" in res.text
    assert res.metadata["full_letter"] == "Dear Principal,\nWe request venue permission."


def test_repeated_ingestion_produces_identical_node_ids():
    """Test 2: Repeated ingestion produces identical node.id_ but without deduplication in PGVectorStore."""
    node1 = TextNode(
        id_="doc_alpha:tab_01",
        text="Body 1",
        metadata={
            "source_doc_id": "doc_alpha:tab_01",
            "letter_type": "FINANCIAL_REQUEST",
            "subject": "Grant",
            "full_markdown": "Body 1",
        },
    )
    node2 = TextNode(
        id_="doc_alpha:tab_01",
        text="Body 1",
        metadata={
            "source_doc_id": "doc_alpha:tab_01",
            "letter_type": "FINANCIAL_REQUEST",
            "subject": "Grant",
            "full_markdown": "Body 1",
        },
    )
    formatter = NodeTextFormatter()
    res1 = formatter([node1])[0]
    res2 = formatter([node2])[0]

    # Both nodes share the same node_id
    assert res1.id_ == res2.id_ == "doc_alpha:tab_01"
    # When passed to PGVectorStore.async_add, each generates an independent session.add()
    # which causes duplicate rows unless explicitly deleted or upserted.


def test_updated_source_content():
    """Test 3: Updated source content updates the composite text and full_letter."""
    updated_node = TextNode(
        id_="doc_alpha:tab_01",
        text="Initial text",
        metadata={
            "source_doc_id": "doc_alpha:tab_01",
            "letter_type": "VENUE_PERMISSION",
            "subject": "Updated Subject for Hall B",
            "description": "Updated purpose",
            "full_markdown": "Updated body text for Hall B.",
        },
    )
    formatter = NodeTextFormatter()
    res = formatter([updated_node])[0]

    assert "Updated Subject for Hall B" in res.text
    assert res.metadata["full_letter"] == "Updated body text for Hall B."


@pytest.mark.asyncio
async def test_complete_extraction_failure_guard():
    """Test 4: When all nodes fail extraction, LegacyTableSync does not wipe tables."""
    failed_node = TextNode(
        id_="doc_fail:tab_01",
        text="Raw text",
        metadata={
            "source_doc_id": "doc_fail:tab_01",
            "tab_title": "Failed Tab",
            # missing letter_type and description
        },
    )
    sync = LegacyTableSync()
    with patch("services.pipeline.pipeline_components.legacy_sync.AsyncSessionLocal") as mock_session_cls:
        out = await sync.acall([failed_node])
        assert len(out) == 1
        # DB session was not entered -> tables were not cleared
        mock_session_cls.assert_not_called()


@pytest.mark.asyncio
async def test_partial_extraction_failure_deletes_global_letter_type_memory():
    """Test 5: Partial extraction failure reveals that LetterTypeMemory is cleared globally."""
    good_node = TextNode(
        id_="doc_mix:tab_01",
        text="Good text",
        metadata={
            "source_doc_id": "doc_mix:tab_01",
            "letter_type": "VENUE_PERMISSION",
            "primary_domain": "INFRASTRUCTURE",
            "description": "Venue request",
            "approval_chain": [],
        },
    )
    failed_node = TextNode(
        id_="doc_mix:tab_02",
        text="Failed text",
        metadata={
            "source_doc_id": "doc_mix:tab_02",
            "tab_title": "Failed Tab 2",
            # missing letter_type and description
        },
    )

    sync = LegacyTableSync()
    mock_session = AsyncMock()

    with patch("services.pipeline.pipeline_components.legacy_sync.AsyncSessionLocal") as mock_session_cls, \
         patch.object(sync, "_generate_description_embeddings", return_value=[[0.1] * 768, None]), \
         patch("services.pipeline.pipeline_components.legacy_sync._load_existing_designations", return_value={}):

        mock_session_cls.return_value.__aenter__.return_value = mock_session

        await sync.acall([good_node, failed_node])

        # Verify that delete was called on LetterTypeMemory globally (showing the wipe risk)
        executed_statements = [str(call.args[0]) for call in mock_session.execute.call_args_list]
        assert any("DELETE FROM letter_type_memory" in s for s in executed_statements)
