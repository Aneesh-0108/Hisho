from sqlalchemy.orm import DeclarativeBase
from datetime import datetime
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    String,
    Text,
    Boolean,
    Integer,
    BigInteger,
    Identity,
    ForeignKey,
    JSON,
    DateTime,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    pass


class LetterType(Base):
    __tablename__ = "letter_type"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    approval_chain: Mapped[dict] = mapped_column(JSON, nullable=False)
    template: Mapped[str] = mapped_column(Text, nullable=False)
    letters = relationship("Letter", back_populates="letter_type")


class Letter(Base):
    __tablename__ = "letters"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    reference_number: Mapped[str] = mapped_column(String, nullable=False)
    letter_type_id: Mapped[str] = mapped_column(
        ForeignKey("letter_type.id"), nullable=False
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    approved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    letter_type = relationship(
        "LetterType",
        back_populates="letters",
    )


class LetterMemory(Base):
    """Past letters used as style/format examples for RAG retrieval."""

    __tablename__ = "letter_memory"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    source_doc_id: Mapped[str | None] = mapped_column(
        String, unique=True, nullable=True
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )


class Official(Base):
    __tablename__ = "officials"
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    address: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    organization: Mapped[str] = mapped_column(Text, nullable=False)
    signature: Mapped[str] = mapped_column(Text, nullable=False)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class Report(Base):
    __tablename__ = "reports"
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    reference_number: Mapped[str] = mapped_column(String, nullable=False)
    report_type: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    approved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class User(Base):
    __tablename__ = "user"
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
    )
    username: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, nullable=False)
    hashed_password: Mapped[str | None] = mapped_column(String, nullable=True)
    full_name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    organization: Mapped[str | None] = mapped_column(Text, nullable=True)
    signature: Mapped[str | None] = mapped_column(Text, nullable=True)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    google_id: Mapped[str | None] = mapped_column(String, nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(String, nullable=True)
    google_access_token: Mapped[str | None] = mapped_column(String, nullable=True)
    google_refresh_token: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class LetterTypeMemory(Base):
    """LLM-extracted letter-type metadata from ingested Google Doc tabs."""

    __tablename__ = "letter_type_memory"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    source_doc_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    letter_type: Mapped[str] = mapped_column(String, nullable=False)
    primary_domain: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    approval_chain: Mapped[dict] = mapped_column(JSON, nullable=False)
    table_info: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)
    template_url: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )


class DataLetterNode(Base):
    """Canonical store of historical reference letters and composite embeddings in PostgreSQL pgvector."""

    __tablename__ = "data_letter_nodes"
    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=False),
        primary_key=True,
    )
    node_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    text: Mapped[str] = mapped_column(String, nullable=False)
    metadata_: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)


class Designation(Base):
    """Unique designations extracted from letter approval chains."""

    __tablename__ = "designations"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    incumbent: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )


class Signature(Base):
    """Signature images extracted from letters."""

    __tablename__ = "signatures"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    source_doc_id: Mapped[str] = mapped_column(String, nullable=False)
    image_url: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
