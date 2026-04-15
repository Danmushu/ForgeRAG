"""
Backfill flow test using SQLiteStore (no real DB) + fake embedder
+ in-memory fake vector store. Verifies the end-to-end contract.
"""

from __future__ import annotations

import uuid

import pytest

from config import RelationalConfig, SQLiteConfig
from embedder.backfill import backfill_embeddings
from persistence.ingestion_writer import IngestionWriter
from persistence.store import Store
from persistence.vector.base import VectorHit, VectorItem

from .test_sqlite_store import _sample_doc  # reuse fixture builder

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEmbedder:
    backend = "fake"
    dimension = 3
    batch_size = 8

    def embed_texts(self, texts):
        return [[float(len(t)), 0.0, 1.0] for t in texts]

    def embed_chunks(self, chunks):
        return {c.chunk_id: [float(len(c.content)), 0.0, 1.0] for c in chunks if c.content.strip()}


class FakeVectorStore:
    backend = "fake"
    dimension = 3

    def __init__(self):
        self.items: dict[str, VectorItem] = {}

    def connect(self):
        pass

    def close(self):
        pass

    def ensure_schema(self):
        pass

    def upsert(self, items):
        for it in items:
            self.items[it.chunk_id] = it

    def delete_chunks(self, chunk_ids):
        for cid in chunk_ids:
            self.items.pop(cid, None)

    def delete_parse_version(self, doc_id, parse_version):
        for cid in list(self.items):
            it = self.items[cid]
            if it.doc_id == doc_id and it.parse_version == parse_version:
                del self.items[cid]

    def search(self, query_vector, *, top_k, filter=None):
        return [VectorHit(chunk_id=cid, score=1.0) for cid in list(self.items)[:top_k]]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rel(tmp_path):
    cfg = RelationalConfig(backend="sqlite", sqlite=SQLiteConfig(path=str(tmp_path / "bf.db")))
    store = Store(cfg)
    store.connect()
    store.ensure_schema(with_vector=False, embedding_dim=3)
    yield store
    store.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBackfill:
    def test_backfill_populates_vector_store(self, rel):
        doc_id = f"t_{uuid.uuid4().hex[:6]}"
        doc, tree, chunks = _sample_doc(doc_id)

        IngestionWriter(rel, vector=None).write(doc, tree, chunks)

        vec = FakeVectorStore()
        emb = FakeEmbedder()
        stats = backfill_embeddings(
            relational=rel,
            vector=vec,
            embedder=emb,
            doc_ids=[doc_id],
        )
        assert stats[doc_id] == 1
        assert len(vec.items) == 1
        item = next(iter(vec.items.values()))
        assert item.doc_id == doc_id
        assert len(item.embedding) == 3

    def test_backfill_missing_doc(self, rel):
        vec = FakeVectorStore()
        stats = backfill_embeddings(
            relational=rel,
            vector=vec,
            embedder=FakeEmbedder(),
            doc_ids=["nope"],
        )
        assert stats == {"nope": 0}
        assert len(vec.items) == 0


class TestWriterInlineHook:
    def test_writer_uses_embedder_when_no_embeddings_passed(self, rel):
        doc_id = f"t_{uuid.uuid4().hex[:6]}"
        doc, tree, chunks = _sample_doc(doc_id)

        vec = FakeVectorStore()
        emb = FakeEmbedder()
        writer = IngestionWriter(rel, vector=vec, embedder=emb)
        writer.write(doc, tree, chunks)

        assert len(vec.items) == 1
        assert next(iter(vec.items.values())).chunk_id == f"{doc_id}:1:c1"

    def test_explicit_embeddings_override_embedder(self, rel):
        doc_id = f"t_{uuid.uuid4().hex[:6]}"
        doc, tree, chunks = _sample_doc(doc_id)

        vec = FakeVectorStore()
        writer = IngestionWriter(rel, vector=vec, embedder=FakeEmbedder())
        writer.write(
            doc,
            tree,
            chunks,
            embeddings={chunks[0].chunk_id: [9.0, 9.0, 9.0]},
        )
        item = vec.items[chunks[0].chunk_id]
        assert item.embedding == [9.0, 9.0, 9.0]
