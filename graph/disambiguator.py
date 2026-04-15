"""
Embedding-based entity disambiguation.

When upserting an entity, checks whether a semantically identical entity
already exists (e.g. "Apple Inc" vs "Apple" vs "AAPL") using cosine
similarity on name embeddings. If a match is found above the threshold,
the new entity's ID is redirected to the existing canonical entity.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from .base import Entity, GraphStore, Relation
from .faiss_index import VectorIndex

if TYPE_CHECKING:
    from embedder.base import Embedder

log = logging.getLogger(__name__)


class EntityDisambiguator:
    """FAISS-accelerated entity name embedding cache for disambiguation."""

    def __init__(
        self,
        embedder: Embedder,
        threshold: float = 0.85,
        candidate_top_k: int = 10,
    ):
        self.embedder = embedder
        self.threshold = threshold
        self._candidate_top_k = candidate_top_k
        self._lock = threading.RLock()
        self._index = VectorIndex()
        # original_entity_id → canonical_entity_id
        self._redirects: dict[str, str] = {}

    def load_existing(self, entities: list[Entity]) -> None:
        """Warm the FAISS index from existing entities that already have embeddings."""
        with self._lock:
            keys, vecs = [], []
            for e in entities:
                if e.name_embedding:
                    keys.append(e.entity_id)
                    vecs.append(e.name_embedding)
            if keys:
                self._index.add_batch(keys, vecs)

    def find_match(self, entity: Entity) -> str | None:
        """Return the canonical entity_id if a similar entity exists, else None."""
        with self._lock:
            if self._index.size == 0:
                return None
            if not entity.name_embedding:
                return None

            hits = self._index.search(entity.name_embedding, self._candidate_top_k)
            for eid, score in hits:
                if eid == entity.entity_id:
                    continue
                if score >= self.threshold:
                    return eid
            return None

    def resolve(self, entity_id: str) -> str:
        """Follow redirect chain to canonical ID."""
        with self._lock:
            return self._redirects.get(entity_id, entity_id)

    def register(self, entity: Entity) -> None:
        """Add entity to the FAISS index after successful upsert."""
        with self._lock:
            if entity.name_embedding:
                self._index.add(entity.entity_id, entity.name_embedding)

    def add_redirect(self, from_id: str, to_id: str) -> None:
        """Record that from_id should be treated as to_id."""
        with self._lock:
            self._redirects[from_id] = to_id


class DisambiguatingGraphStore:
    """
    Wrapper that adds entity disambiguation to any GraphStore.

    Intercepts upsert_entity / upsert_relation calls to check for
    semantic duplicates. Everything else is delegated to the inner store.
    """

    def __init__(
        self,
        inner: GraphStore,
        disambiguator: EntityDisambiguator,
    ):
        self._inner = inner
        self._dis = disambiguator

    def upsert_entity(self, entity: Entity) -> None:
        # Embed name if not already done
        if not entity.name_embedding:
            try:
                vecs = self._dis.embedder.embed_texts([entity.name])
                entity.name_embedding = vecs[0]
            except Exception:
                log.warning("Failed to embed entity name %r", entity.name)

        # Check for existing match
        match_id = self._dis.find_match(entity)
        if match_id and match_id != entity.entity_id:
            log.info(
                "Disambiguated entity %r (id=%s) → existing %s",
                entity.name,
                entity.entity_id,
                match_id,
            )
            original_id = entity.entity_id
            entity.entity_id = match_id
            self._dis.add_redirect(original_id, match_id)

        self._inner.upsert_entity(entity)
        self._dis.register(entity)

    def upsert_relation(self, relation: Relation) -> None:
        # Resolve any redirected entity IDs
        new_src = self._dis.resolve(relation.source_entity)
        new_tgt = self._dis.resolve(relation.target_entity)
        if new_src != relation.source_entity or new_tgt != relation.target_entity:
            relation.source_entity = new_src
            relation.target_entity = new_tgt
            relation.relation_id = f"{new_src}->{new_tgt}"
        self._inner.upsert_relation(relation)

    def __getattr__(self, name):
        """Delegate all other methods to the inner store."""
        return getattr(self._inner, name)
