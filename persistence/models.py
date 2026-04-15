"""
SQLAlchemy 2.0 declarative models.

Portable across Postgres / MySQL / SQLite via the dialect URL.
Array-like fields use the JSON column type (JSONB on Postgres,
JSON on MySQL, TEXT on SQLite) because our access pattern is
"store a list, read it back whole" rather than indexed lookups.

Hard-overwrite versioning:
    (doc_id, parse_version) is treated as the unit of truth.
    ingestion_writer deletes old rows by parse_version inside a
    single transaction, then inserts the new ones.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Files (new: content-addressed user uploads)
# ---------------------------------------------------------------------------


class File(Base):
    __tablename__ = "files"

    file_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(128), index=True)
    storage_key: Mapped[str] = mapped_column(String(512))
    original_name: Mapped[str] = mapped_column(String(512))
    display_name: Mapped[str] = mapped_column(String(512))
    size_bytes: Mapped[int] = mapped_column(Integer)
    mime_type: Mapped[str] = mapped_column(String(128))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    file_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("files.file_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # If the source was converted to PDF (e.g. DOCX→PDF), this points
    # to the converted PDF file for viewing/highlighting.
    pdf_file_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("files.file_id", ondelete="SET NULL"),
        nullable=True,
    )
    filename: Mapped[str] = mapped_column(String(512), default="")
    format: Mapped[str] = mapped_column(String(32))
    active_parse_version: Mapped[int] = mapped_column(Integer, default=1)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    doc_profile_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    parse_trace_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # --- Processing status ---
    status: Mapped[str | None] = mapped_column(String(32), nullable=True, server_default="pending")
    # Embedding
    embed_status: Mapped[str | None] = mapped_column(String(32), nullable=True, server_default="pending")
    embed_provider_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embed_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    embed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # LLM enrichment
    enrich_status: Mapped[str | None] = mapped_column(String(32), nullable=True, server_default="pending")
    enrich_provider_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    enrich_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    enrich_summary_count: Mapped[int | None] = mapped_column(Integer, nullable=True, server_default="0")
    enrich_image_count: Mapped[int | None] = mapped_column(Integer, nullable=True, server_default="0")
    enrich_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Per-phase timing
    parse_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    parse_completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    structure_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    structure_completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    enrich_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    embed_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Knowledge Graph extraction
    kg_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    kg_entity_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    kg_relation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    kg_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    kg_completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    kg_provider_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    kg_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Tree navigation eligibility
    tree_navigable: Mapped[bool | None] = mapped_column(Boolean, nullable=True, server_default="1")
    tree_quality: Mapped[float | None] = mapped_column(Float, nullable=True)
    tree_method: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Error message (human-readable reason for status="error")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Page dimensions: [{page_no, width, height}, ...]
    pages_json: Mapped[list | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
    )


# ---------------------------------------------------------------------------
# Parsed blocks
# ---------------------------------------------------------------------------


class ParsedBlock(Base):
    __tablename__ = "parsed_blocks"

    block_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("documents.doc_id", ondelete="CASCADE"),
        index=True,
    )
    parse_version: Mapped[int] = mapped_column(Integer, index=True)
    page_no: Mapped[int] = mapped_column(Integer, index=True)
    seq: Mapped[int] = mapped_column(Integer)

    bbox_x0: Mapped[float] = mapped_column(Float)
    bbox_y0: Mapped[float] = mapped_column(Float)
    bbox_x1: Mapped[float] = mapped_column(Float)
    bbox_y1: Mapped[float] = mapped_column(Float)

    type: Mapped[str] = mapped_column(String(32))
    level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)

    table_html: Mapped[str | None] = mapped_column(Text, nullable=True)
    table_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    figure_storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    figure_mime: Mapped[str | None] = mapped_column(String(64), nullable=True)
    figure_caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    formula_latex: Mapped[str | None] = mapped_column(Text, nullable=True)

    excluded: Mapped[bool] = mapped_column(Boolean, default=False)
    excluded_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    caption_of: Mapped[str | None] = mapped_column(String(255), nullable=True)
    cross_ref_targets: Mapped[list] = mapped_column(JSON, default=list)

    __table_args__ = (Index("ix_parsed_blocks_doc_version", "doc_id", "parse_version"),)


# ---------------------------------------------------------------------------
# Doc tree (pure JSONB)
# ---------------------------------------------------------------------------


class DocTreeRow(Base):
    __tablename__ = "doc_trees"

    doc_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("documents.doc_id", ondelete="CASCADE"),
    )
    parse_version: Mapped[int] = mapped_column(Integer)
    root_id: Mapped[str] = mapped_column(String(255))
    quality_score: Mapped[float] = mapped_column(Float)
    generation_method: Mapped[str] = mapped_column(String(32))
    tree_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (PrimaryKeyConstraint("doc_id", "parse_version"),)


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Query traces (retrieval pipeline audit log)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Runtime settings (frontend-editable config overrides)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Conversations (multi-turn chat)
# ---------------------------------------------------------------------------


class Conversation(Base):
    __tablename__ = "conversations"

    conversation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class Message(Base):
    __tablename__ = "messages"

    message_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        index=True,
    )
    role: Mapped[str] = mapped_column(String(16))  # user / assistant
    content: Mapped[str] = mapped_column(Text)
    # For assistant messages: link back to the trace + citations
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    citations_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# Runtime settings (frontend-editable config overrides)
# ---------------------------------------------------------------------------


class Setting(Base):
    """
    Key-value store for runtime config overrides.

    The yaml file is the base config. Settings in this table are
    OVERLAID on top — a key like "retrieval.rerank.enabled" with
    value_json=true means "override that one field at runtime".

    Groups allow the frontend to organize settings into tabs:
        llm, embedding, retrieval, parsing, system

    Schema is intentionally flat (not nested JSONB) so the frontend
    can render a simple form: one row per toggle/input.
    """

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    value_json: Mapped[Any] = mapped_column(JSON)
    group_name: Mapped[str] = mapped_column(String(64), index=True, default="system")
    label: Mapped[str] = mapped_column(String(255), default="")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    value_type: Mapped[str] = mapped_column(String(32), default="string")  # string/int/float/bool/enum/secret
    enum_options: Mapped[list | None] = mapped_column(JSON, nullable=True)  # for value_type=enum
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# LLM Providers (pluggable model registry)
# ---------------------------------------------------------------------------


class LLMProvider(Base):
    """
    Registry of LLM / embedding / reranker endpoints.

    Allows the system to reference models by a short user-defined name
    instead of hardcoding model strings and API keys in settings.
    The `provider_type` column distinguishes chat models from embedding
    models from rerankers so the UI can filter appropriately.
    """

    __tablename__ = "llm_providers"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    provider_type: Mapped[str] = mapped_column(String(32), index=True)  # chat / embedding / reranker
    api_base: Mapped[str | None] = mapped_column(String(512), nullable=True)
    model_name: Mapped[str] = mapped_column(String(255))  # litellm model string
    api_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# Query traces (retrieval pipeline audit log)
# ---------------------------------------------------------------------------


class QueryTrace(Base):
    __tablename__ = "query_traces"

    trace_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    total_ms: Mapped[int] = mapped_column(Integer, default=0)
    total_llm_ms: Mapped[int] = mapped_column(Integer, default=0)
    total_llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    finish_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    citations_used: Mapped[list] = mapped_column(JSON, default=list)
    trace_json: Mapped[dict] = mapped_column(JSON)  # full trace phases
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


class ChunkRow(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("documents.doc_id", ondelete="CASCADE"),
        index=True,
    )
    parse_version: Mapped[int] = mapped_column(Integer, index=True)
    node_id: Mapped[str] = mapped_column(String(255), index=True)

    content: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(String(32))
    block_ids: Mapped[list] = mapped_column(JSON, default=list)
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer)
    y_sort: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")

    section_path: Mapped[list] = mapped_column(JSON, default=list)
    ancestor_node_ids: Mapped[list] = mapped_column(JSON, default=list)
    cross_ref_chunk_ids: Mapped[list] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (Index("ix_chunks_doc_version", "doc_id", "parse_version"),)
