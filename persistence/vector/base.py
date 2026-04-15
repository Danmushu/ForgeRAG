"""
Vector store protocol and factory.

Backends:
    - pgvector: embeddings live in the same Postgres database as
                metadata, inside chunks.embedding.
    - chromadb: embeddings live in a separate ChromaDB collection,
                keyed by chunk_id.

In both cases the chunk_id is the foreign key that ties the
vector store back to the relational store. Metadata mirrored into
the vector store is limited to what's useful for pre-filtering:
doc_id, parse_version, node_id, content_type, page range.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from config import VectorConfig


@dataclass
class VectorItem:
    chunk_id: str
    doc_id: str
    parse_version: int
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorHit:
    chunk_id: str
    score: float  # similarity, higher = better
    doc_id: str | None = None
    parse_version: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    backend: str  # "pgvector" | "chromadb" | "qdrant" | "milvus" | "weaviate"
    dimension: int

    def connect(self) -> None: ...
    def close(self) -> None: ...
    def ensure_schema(self) -> None: ...

    def upsert(self, items: list[VectorItem]) -> None: ...

    def delete_chunks(self, chunk_ids: list[str]) -> None: ...

    def delete_parse_version(self, doc_id: str, parse_version: int) -> None: ...

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorHit]: ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_vector_store(
    cfg: VectorConfig,
    *,
    relational_store: Any | None = None,  # needed by pgvector
) -> VectorStore:
    if cfg.backend == "pgvector":
        from .pgvector import PgvectorStore

        assert cfg.pgvector is not None
        if relational_store is None or getattr(relational_store, "backend", None) != "postgres":
            raise ValueError("pgvector backend requires a connected Store with backend=postgres")
        return PgvectorStore(cfg.pgvector, relational_store)

    if cfg.backend == "chromadb":
        from .chroma import ChromaStore

        assert cfg.chromadb is not None
        return ChromaStore(cfg.chromadb)

    if cfg.backend == "qdrant":
        from .qdrant import QdrantStore

        assert cfg.qdrant is not None
        return QdrantStore(cfg.qdrant)

    if cfg.backend == "milvus":
        from .milvus import MilvusStore

        assert cfg.milvus is not None
        return MilvusStore(cfg.milvus)

    if cfg.backend == "weaviate":
        from .weaviate import WeaviateStore

        assert cfg.weaviate is not None
        return WeaviateStore(cfg.weaviate)

    raise ValueError(f"unknown vector backend: {cfg.backend!r}")
