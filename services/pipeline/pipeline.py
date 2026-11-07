"""Assembly and execution of the LlamaIndex document ingestion pipeline.

Pipeline stages (in execution order):
1. GoogleDocJsonParser: Converts raw JSON to Markdown.
2. LetterMetadataExtractor: Uses the configured extraction model to extract
   structured data, populates node.metadata, and rewrites node.text to the
   generalized description.
3. LegacyTableSync: Syncs to Signature, Designation, and LetterTypeMemory
   tables (generates description-based embeddings for LetterTypeMemory).
4. NodeTextFormatter: Restructures metadata and sets node.text to subject.
5. OllamaEmbedding: Generates 768-dim vectors from subject (node.text).
6. PGVectorStore: Inserts into the PostgreSQL letter_nodes table.
"""

import logging
from typing import Sequence

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline

from services.pipeline.pipeline_components import (
    GoogleDocJsonParser,
    LetterMetadataExtractor,
    LegacyTableSync,
    NodeTextFormatter,
    get_embed_model,
    get_pgvector_store,
)

logger = logging.getLogger(__name__)


def build_pipeline() -> IngestionPipeline:
    """Assemble the LlamaIndex IngestionPipeline."""
    return IngestionPipeline(
        transformations=[
            GoogleDocJsonParser(),
            LetterMetadataExtractor(),
            LegacyTableSync(),
            NodeTextFormatter(require_category=True),
            get_embed_model(),
        ],
        vector_store=get_pgvector_store(),
    )


async def run_ingestion(documents: Sequence[Document]) -> list:
    """Run the ingestion pipeline on a set of documents asynchronously.

    Args:
        documents: A list of LlamaIndex Document objects. The `text` property
                   must contain the raw JSON string from DocumentTransformer.

    Returns:
        The processed nodes that were successfully ingested.
    """
    pipeline = build_pipeline()
    logger.info(f"Starting ingestion pipeline for {len(documents)} documents...")

    nodes = await pipeline.arun(documents=documents)

    logger.info(f"Ingestion pipeline completed. Ingested {len(nodes)} nodes.")
    return nodes
