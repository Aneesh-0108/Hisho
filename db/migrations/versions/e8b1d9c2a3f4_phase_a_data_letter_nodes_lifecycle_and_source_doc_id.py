"""phase_a_data_letter_nodes_lifecycle_and_source_doc_id

Revision ID: e8b1d9c2a3f4
Revises: 200be0292be9
Create Date: 2026-11-01 20:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = 'e8b1d9c2a3f4'
down_revision: Union[str, Sequence[str], None] = '200be0292be9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def check_data_integrity(conn) -> None:
    """Run defensive checks on existing data_letter_nodes before applying constraints."""
    # Check 1: Null node_ids
    null_count = conn.execute(
        sa.text("SELECT count(*) FROM data_letter_nodes WHERE node_id IS NULL;")
    ).scalar()
    if null_count and null_count > 0:
        raise RuntimeError(f"Cannot apply NOT NULL: {null_count} rows have NULL node_id.")

    # Check 2: Duplicate node_ids
    dup_count = conn.execute(sa.text("""
        SELECT count(*) FROM (
            SELECT node_id FROM data_letter_nodes
            GROUP BY node_id HAVING count(*) > 1
        ) dups;
    """)).scalar()
    if dup_count and dup_count > 0:
        raise RuntimeError(
            f"Cannot apply UNIQUE constraint: {dup_count} duplicate node_ids exist. "
            "Execute legacy cleanup (Phase E) before applying constraints."
        )


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    has_nodes_table = inspector.has_table("data_letter_nodes")

    # 1. Ensure pgvector extension is available
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. Lifecycle branch for data_letter_nodes
    if not has_nodes_table:
        # Path A: Fresh installation
        op.create_table(
            "data_letter_nodes",
            sa.Column("id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
            sa.Column("node_id", sa.String(), nullable=False),
            sa.Column("text", sa.String(), nullable=False),
            sa.Column("metadata_", sa.JSON(), nullable=True),
            sa.Column("embedding", Vector(768), nullable=True),
            sa.UniqueConstraint("node_id", name="uq_data_letter_nodes_node_id"),
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS data_letter_nodes_embedding_hnsw_idx "
            "ON data_letter_nodes USING hnsw (embedding vector_cosine_ops);"
        )
    else:
        # Path B: Existing instance
        check_data_integrity(conn)
        op.alter_column("data_letter_nodes", "node_id", nullable=False)

        existing_constraints = [c["name"] for c in inspector.get_unique_constraints("data_letter_nodes")]
        if "uq_data_letter_nodes_node_id" not in existing_constraints:
            op.create_unique_constraint(
                "uq_data_letter_nodes_node_id", "data_letter_nodes", ["node_id"]
            )

        op.execute(
            "CREATE INDEX IF NOT EXISTS data_letter_nodes_embedding_hnsw_idx "
            "ON data_letter_nodes USING hnsw (embedding vector_cosine_ops);"
        )

    # 3. letter_type_memory source_doc_id provenance column
    if inspector.has_table("letter_type_memory"):
        columns = [c["name"] for c in inspector.get_columns("letter_type_memory")]
        if "source_doc_id" not in columns:
            op.add_column(
                "letter_type_memory",
                sa.Column("source_doc_id", sa.String(), nullable=True)
            )
            op.create_index(
                "ix_letter_type_memory_source_doc_id",
                "letter_type_memory",
                ["source_doc_id"]
            )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    # 1. Revert letter_type_memory source_doc_id column
    if inspector.has_table("letter_type_memory"):
        columns = [c["name"] for c in inspector.get_columns("letter_type_memory")]
        if "source_doc_id" in columns:
            op.drop_index("ix_letter_type_memory_source_doc_id", table_name="letter_type_memory")
            op.drop_column("letter_type_memory", "source_doc_id")

    # 2. Revert data_letter_nodes constraints
    if inspector.has_table("data_letter_nodes"):
        existing_constraints = [c["name"] for c in inspector.get_unique_constraints("data_letter_nodes")]
        if "uq_data_letter_nodes_node_id" in existing_constraints:
            op.drop_constraint("uq_data_letter_nodes_node_id", "data_letter_nodes", type_="unique")
        op.alter_column("data_letter_nodes", "node_id", nullable=True)
