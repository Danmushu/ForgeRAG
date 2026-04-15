"""
Backfill embeddings for previously-ingested documents.

Use cases:
    - Changed embedding model: reindex all chunks for a set of docs
    - Ingestion flow skipped embeddings (e.g. embedder was offline)
    - Restoring vectors after a vector store wipe

The backfill walks the relational store for the active parse_version
of each doc_id, rebuilds Chunk dataclasses via the persistence.serde
reader, calls embedder.embed_chunks(), then upserts into the vector
store.

This function does not touch the relational store's chunks / trees
/ blocks -- it only populates embeddings. Safe to rerun.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from persistence.serde import row_to_chunk
from persistence.store import Store as RelationalStore
from persistence.vector.base import VectorItem, VectorStore

from .base import Embedder

log = logging.getLogger(__name__)


def backfill_embeddings(
    *,
    relational: RelationalStore,
    vector: VectorStore,
    embedder: Embedder,
    doc_ids: Iterable[str],
    parse_version: int | None = None,
) -> dict[str, int]:
    """
    Backfill embeddings for the given doc_ids.

    Args:
        parse_version: if None, the backfill reads each doc's
            `active_parse_version` from the documents table; pass a
            specific int to force backfilling a particular version.

    Returns a dict {doc_id: number_of_chunks_embedded}.
    """
    stats: dict[str, int] = {}
    for doc_id in doc_ids:
        pv = parse_version
        if pv is None:
            row = relational.get_document(doc_id)
            if not row:
                log.warning("backfill: doc %s not found", doc_id)
                stats[doc_id] = 0
                continue
            pv = row["active_parse_version"]

        chunk_rows = relational.get_chunks(doc_id, pv)
        if not chunk_rows:
            stats[doc_id] = 0
            continue
        chunks = [row_to_chunk(r) for r in chunk_rows]

        embeddings = embedder.embed_chunks(chunks)
        items = [
            VectorItem(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                parse_version=c.parse_version,
                embedding=embeddings[c.chunk_id],
                metadata={
                    "node_id": c.node_id,
                    "content_type": c.content_type,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                },
            )
            for c in chunks
            if c.chunk_id in embeddings
        ]
        vector.upsert(items)
        stats[doc_id] = len(items)
        log.info("backfill doc=%s version=%d embedded=%d", doc_id, pv, len(items))
    return stats
