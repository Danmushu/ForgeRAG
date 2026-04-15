"""
Vector retrieval path.

Thin wrapper around VectorStore.search(). Kept as its own module
so the pipeline's orchestration stays readable and so that we can
add query-vector caching / query-expansion hooks here without
polluting the VectorStore protocol.
"""

from __future__ import annotations

from persistence.vector.base import VectorStore

from .types import ScoredChunk


def search_vectors(
    vector_store: VectorStore,
    query_embedding: list[float],
    *,
    top_k: int,
    filter: dict | None = None,
) -> list[ScoredChunk]:
    hits = vector_store.search(query_embedding, top_k=top_k, filter=filter)
    return [ScoredChunk(chunk_id=h.chunk_id, score=h.score, source="vector") for h in hits]
