"""Integration tests for pipeline transformations (parser + formatter)."""

import json
from llama_index.core.schema import Document
import pytest

from services.pipeline.pipeline_components.gdoc_parser import GoogleDocJsonParser
from services.pipeline.pipeline_components.node_formatter import NodeTextFormatter


def test_pipeline_transformation_preserves_letter_and_builds_composite_text():
    """Verify end-to-end transformation from raw JSON document through parser and formatter."""
    raw_elements = [
        {"type": "text", "content": "SCEPTIX CLUB\nNational Institute of Technology"},
        {"type": "text", "content": "Ref: SCEPTIX/2026/01"},
        {"type": "text", "content": "To,\nThe Dean (Student Welfare)"},
        {"type": "text", "content": "Subject: Request for Collaborative Funds for Winter Hackathon"},
        {"type": "text", "content": "Respected Sir,\nWe request sanction of Rs 50,000 for event prizes."},
        {"type": "table", "rows": [["Item", "Estimated Cost"], ["Prizes", "Rs 30,000"], ["Logistics", "Rs 20,000"]]},
        {"type": "text", "content": "Sincerely,\nSecretary, Sceptix Club"},
    ]
    raw_json = json.dumps(raw_elements)

    doc = Document(
        text=raw_json,
        metadata={
            "source_doc_id": "google_doc_sceptix_01:tab_hackathon",
            "tab_title": "Winter Hackathon",
        },
    )

    # 1. Parse JSON -> TextNode
    parser = GoogleDocJsonParser()
    parsed_nodes = parser.get_nodes_from_documents([doc])
    assert len(parsed_nodes) == 1
    node = parsed_nodes[0]

    # Verify parser set stable ID and full markdown
    assert node.id_ == "google_doc_sceptix_01:tab_hackathon"
    assert "SCEPTIX CLUB" in node.metadata["full_markdown"]
    assert "Prizes" in node.metadata["full_markdown"]

    # 2. Simulate extraction metadata (as populated by LetterMetadataExtractor)
    node.metadata.update({
        "ref_no": "SCEPTIX/2026/01",
        "date": "12/03/2026",
        "subject": "Request for Collaborative Funds for Winter Hackathon",
        "letter_type": "FINANCIAL_REQUEST",
        "primary_domain": "FINANCE",
        "description": "Formal request seeking release of collaborative funds for event arrangements and prizes.",
        "table_info": {
            "summary": "Budget breakdown",
            "category": "FINANCIAL",
            "headers": ["Item", "Estimated Cost"],
        },
        "approval_chain": [
            {"step_number": 1, "designation_title": "Faculty Advisor", "incumbent": "Dr. Smith"},
            {"step_number": 2, "designation_title": "Dean (Student Welfare)", "incumbent": "Dr. Jones"},
        ],
    })

    # 3. Format node text & finalize metadata
    formatter = NodeTextFormatter()
    formatted_nodes = formatter([node])
    final_node = formatted_nodes[0]

    # Check node ID stability
    assert final_node.id_ == "google_doc_sceptix_01:tab_hackathon"

    # Check composite text formulation
    assert "Type: FINANCIAL_REQUEST" in final_node.text
    assert "Subject: Request for Collaborative Funds for Winter Hackathon" in final_node.text
    assert "Purpose: Formal request seeking release of collaborative funds" in final_node.text
    assert "Content:\nSCEPTIX CLUB" in final_node.text

    # Check full letter preservation in metadata
    full_letter = final_node.metadata["full_letter"]
    assert "SCEPTIX CLUB" in full_letter
    assert "Ref: SCEPTIX/2026/01" in full_letter
    assert "Prizes" in full_letter
    assert "Sincerely" in full_letter

    # Check all extracted fields survived in metadata
    meta = final_node.metadata
    assert meta["letter_type"] == "FINANCIAL_REQUEST"
    assert meta["primary_domain"] == "FINANCE"
    assert meta["source_doc_id"] == "google_doc_sceptix_01:tab_hackathon"
    assert meta["tab_title"] == "Winter Hackathon"
    assert len(meta["approval_chain"]) == 2
    assert meta["approval_chain"][0]["incumbent"] == "Dr. Smith"
