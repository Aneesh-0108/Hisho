"""Tests for GoogleDocJsonParser: stable ID and metadata preservation."""

import json
from llama_index.core.schema import Document
import pytest

from services.pipeline.pipeline_components.gdoc_parser import GoogleDocJsonParser


def test_gdoc_parser_preserves_stable_id_from_source_doc_id():
    """Verify that TextNode id_ is stably derived from source_doc_id."""
    payload = json.dumps([
        {"type": "text", "content": "Sample letter paragraph."}
    ])
    doc = Document(
        text=payload,
        metadata={
            "source_doc_id": "gdoc_101:tab_202",
            "tab_title": "Budget Letter",
        },
    )

    parser = GoogleDocJsonParser()
    nodes = parser.get_nodes_from_documents([doc])

    assert len(nodes) == 1
    node = nodes[0]
    assert node.id_ == "gdoc_101:tab_202"
    assert node.metadata["source_doc_id"] == "gdoc_101:tab_202"
    assert node.metadata["tab_title"] == "Budget Letter"
    assert "Sample letter paragraph." in node.text
    assert node.metadata["full_markdown"] == "Sample letter paragraph."


def test_gdoc_parser_handles_non_json_safely():
    """Verify non-JSON payload passes through without crashing."""
    doc = Document(
        text="Not valid JSON content",
        metadata={"source_doc_id": "raw_doc_1"},
    )
    parser = GoogleDocJsonParser()
    nodes = parser.get_nodes_from_documents([doc])

    assert len(nodes) == 1
    assert nodes[0].get_content() == "Not valid JSON content"
