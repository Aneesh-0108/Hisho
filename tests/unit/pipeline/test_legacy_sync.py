"""Tests for LegacyTableSync: failure handling and non-destructive behavior."""

from unittest.mock import AsyncMock, MagicMock, patch
from llama_index.core.schema import TextNode
import pytest

from services.pipeline.pipeline_components.legacy_sync import (
    LegacyTableSync,
    _sync_to_legacy_tables,
)


def test_sync_to_legacy_tables_does_not_mutate_approval_chain():
    """Verify that _sync_to_legacy_tables does not mutate step dicts in approval_chain."""
    approval_chain = [
        {"step_number": 1, "designation_title": "Faculty Advisor", "incumbent": "Dr. Smith"},
        {"step_number": 2, "designation_title": "Head of Department", "incumbent": "Dr. Kumar"},
    ]
    node = TextNode(
        id_="node_1",
        text="text",
        metadata={
            "letter_type": "FINANCIAL_REQUEST",
            "primary_domain": "FINANCE",
            "description": "Funding request",
            "approval_chain": approval_chain,
            "source_doc_id": "doc1:tab1",
        },
    )

    mock_session = MagicMock()
    existing_codes = {}
    description_embeddings = [[0.1] * 768]

    added, designations_added = _sync_to_legacy_tables(
        mock_session, [node], existing_codes, description_embeddings
    )

    assert added == 1
    assert designations_added == 2

    # Verify incumbent is STILL present in approval_chain (not popped!)
    assert approval_chain[0]["incumbent"] == "Dr. Smith"
    assert approval_chain[1]["incumbent"] == "Dr. Kumar"


@pytest.mark.asyncio
async def test_legacy_sync_skips_when_extraction_failed_for_all_nodes():
    """Verify that when extraction failed for all nodes, acall does not clear tables."""
    node = TextNode(
        id_="node_fail",
        text="raw unextracted text",
        metadata={
            "source_doc_id": "doc_fail:tab1",
            "tab_title": "Failed Tab",
            # No letter_type or description
        },
    )

    sync_component = LegacyTableSync()

    with patch("services.pipeline.pipeline_components.legacy_sync.AsyncSessionLocal") as mock_session_cls:
        result = await sync_component.acall([node])

        # Nodes returned unchanged
        assert len(result) == 1
        # AsyncSessionLocal must NEVER be called, preventing destructive deletions
        mock_session_cls.assert_not_called()
