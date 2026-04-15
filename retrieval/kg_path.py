"""
Knowledge Graph retrieval path.

Multi-level retrieval inspired by LightRAG:

  1. **Local**: Extract entities from query -> graph.get_neighbors()
     -> collect source chunk_ids from related entities/relations
  2. **Global**: Extract keywords from query -> graph.search_entities()
     -> high-level relationship traversal -> chunk_ids
  3. **Community**: Embed query -> cosine search over community summaries
     -> collect chunk_ids from matched community members
  4. **Relation semantic**: Embed query -> cosine search over relation
     description embeddings -> collect chunk_ids

All levels produce scored chunks that are fed into the 4-way RRF merge.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.retrieval import KGPathConfig
    from embedder.base import Embedder
    from graph.base import GraphStore
    from persistence.store import Store

from .types import KGContext, ScoredChunk

log = logging.getLogger(__name__)


class KGPath:
    """Knowledge graph retrieval path."""

    def __init__(
        self,
        cfg: KGPathConfig,
        graph: GraphStore,
        relational: Store,
        *,
        extractor=None,  # KGExtractor instance for query entity extraction
        embedder: Embedder | None = None,
    ):
        self.cfg = cfg
        self.graph = graph
        self.rel = relational
        self.extractor = extractor
        self.embedder = embedder
        self._llm_calls: list[dict] = []  # collect LLM call info for trace
        self.kg_context: KGContext = KGContext()  # synthesized context for prompt injection

    def search(self, query: str) -> list[ScoredChunk]:
        """
        Multi-level KG retrieval.

        Returns a ranked list of ScoredChunks sourced from the
        knowledge graph, ready for RRF merge with other paths.

        Also populates ``self.kg_context`` with entity descriptions,
        relation descriptions, and community summaries — synthesized
        knowledge that the answering layer injects into the LLM prompt
        alongside raw text chunks (inspired by LightRAG).
        """
        self.kg_context = KGContext()

        # Step 1: Extract entities and keywords from query
        entity_names, keywords = self._extract_query_entities(query)
        if not entity_names and not keywords:
            log.debug("KG path: no entities/keywords extracted from query")
            return []

        # Step 2: Local retrieval -- entity neighborhood traversal
        local_chunks = self._local_retrieval(entity_names)

        # Step 3: Global retrieval -- keyword-based entity search
        global_chunks = self._global_retrieval(keywords or entity_names)

        # Step 4: Community retrieval -- semantic search over community summaries
        community_chunks = self._community_retrieval(query)

        # Step 5: Relation semantic search
        relation_chunks = self._relation_retrieval(query)

        # Step 6: Weighted merge
        merged = self._merge_scores(
            local_chunks,
            global_chunks,
            community_chunks,
            relation_chunks,
        )

        # Step 7: Verify chunks exist and return top-k
        verified = self._verify_chunks(merged)
        top_k = sorted(verified, key=lambda s: -s.score)[: self.cfg.top_k]

        log.info(
            "KG path: entities=%d keywords=%d local=%d global=%d "
            "community=%d relation=%d merged=%d top_k=%d "
            "kg_ctx(ent=%d rel=%d comm=%d)",
            len(entity_names),
            len(keywords),
            len(local_chunks),
            len(global_chunks),
            len(community_chunks),
            len(relation_chunks),
            len(merged),
            len(top_k),
            len(self.kg_context.entities),
            len(self.kg_context.relations),
            len(self.kg_context.community_summaries),
        )
        return top_k

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _extract_query_entities(self, query: str) -> tuple[list[str], list[str]]:
        """Extract entity names and keywords from the query using LLM."""
        import time as _time

        t0 = _time.time()
        try:
            if self.extractor is not None:
                result = self.extractor.extract_query_entities(query)
            else:
                # Use kg_path config; if api_key/api_base are empty, log and skip
                if not self.cfg.api_key and not self.cfg.api_base:
                    log.warning(
                        "KG path: no api_key or api_base configured "
                        "(provider_id=%s) — skipping query entity extraction",
                        getattr(self.cfg, "provider_id", None),
                    )
                    return [], []
                from ingestion.kg_extractor import KGExtractor

                ext = KGExtractor(
                    model=self.cfg.model,
                    api_key=self.cfg.api_key,
                    api_key_env=self.cfg.api_key_env,
                    api_base=self.cfg.api_base,
                )
                result = ext.extract_query_entities(query)
            ms = int((_time.time() - t0) * 1000)
            self._llm_calls.append(
                dict(
                    model=getattr(self.cfg, "model", "unknown"),
                    purpose="kg_entity_extraction",
                    latency_ms=ms,
                    output_preview=str(result[0][:5]) if result[0] else "[]",
                )
            )
            return result
        except Exception as e:
            log.warning("KG query entity extraction failed: %s", e)
            return [], []

    def _local_retrieval(
        self,
        entity_names: list[str],
    ) -> dict[str, float]:
        """
        Local retrieval: find entities by name -> traverse neighbors
        -> collect chunk_ids with scores based on hop distance.

        Also collects entity descriptions and relation descriptions
        into ``self.kg_context`` for synthesized context injection.
        """
        from graph.base import entity_id_from_name

        chunk_scores: dict[str, float] = {}
        _seen_entities: set[str] = set()
        _seen_relations: set[str] = set()
        # Cache entity_id → name to avoid repeated get_entity calls
        # (important for Neo4j where each call is a DB round-trip)
        _name_cache: dict[str, str] = {}

        def _resolve_name(eid_: str) -> str:
            if eid_ in _name_cache:
                return _name_cache[eid_]
            ent_ = self.graph.get_entity(eid_)
            name_ = ent_.name if ent_ else eid_
            _name_cache[eid_] = name_
            return name_

        for name in entity_names:
            eid = entity_id_from_name(name)
            entity = self.graph.get_entity(eid)
            if entity is None:
                # Try fuzzy search
                candidates = self.graph.search_entities(name, top_k=3)
                if not candidates:
                    continue
                entity = candidates[0]
                eid = entity.entity_id

            _name_cache[eid] = entity.name

            # Collect entity description for KG context
            if eid not in _seen_entities and entity.description:
                _seen_entities.add(eid)
                self.kg_context.entities.append(
                    {
                        "name": entity.name,
                        "type": entity.entity_type,
                        "description": entity.description,
                        "_eid": eid,
                    }
                )

            # Direct entity chunks (hop 0) -- highest score
            for cid in entity.source_chunk_ids:
                chunk_scores[cid] = max(chunk_scores.get(cid, 0), 1.0)

            # Relations of this entity
            relations = self.graph.get_relations(eid)
            for rel in relations:
                score = 0.8 * rel.weight  # relation weight matters
                for cid in rel.source_chunk_ids:
                    chunk_scores[cid] = max(chunk_scores.get(cid, 0), score)

                # Collect relation description for KG context
                rid = rel.relation_id
                if rid not in _seen_relations and rel.description:
                    _seen_relations.add(rid)
                    self.kg_context.relations.append(
                        {
                            "source": _resolve_name(rel.source_entity),
                            "target": _resolve_name(rel.target_entity),
                            "keywords": rel.keywords,
                            "description": rel.description,
                            "_rid": rid,
                        }
                    )

            # Neighbor entities (hop 1+)
            # get_neighbors returns a flat list without per-item hop info,
            # so we use a uniform decay based on the configured max_hops.
            neighbors = self.graph.get_neighbors(eid, max_hops=self.cfg.max_hops)
            neighbor_score = 1.0 / (1 + self.cfg.max_hops)  # average decay
            for neighbor in neighbors:
                _name_cache[neighbor.entity_id] = neighbor.name
                # Collect neighbor entity descriptions too
                if neighbor.entity_id not in _seen_entities and neighbor.description:
                    _seen_entities.add(neighbor.entity_id)
                    self.kg_context.entities.append(
                        {
                            "name": neighbor.name,
                            "type": neighbor.entity_type,
                            "description": neighbor.description,
                            "_eid": neighbor.entity_id,
                        }
                    )
                for cid in neighbor.source_chunk_ids:
                    chunk_scores[cid] = max(chunk_scores.get(cid, 0), neighbor_score)

        return chunk_scores

    def _global_retrieval(
        self,
        keywords: list[str],
    ) -> dict[str, float]:
        """
        Global retrieval: search entities by keywords -> collect
        chunk_ids from matched entities and their relations.

        Also collects entity descriptions into ``self.kg_context``
        (deduped against those already captured by local retrieval).
        """
        chunk_scores: dict[str, float] = {}
        # Build sets of already-seen IDs from local retrieval for dedup
        _seen_eids = {e.get("_eid") for e in self.kg_context.entities if e.get("_eid")}
        _seen_rids = {r.get("_rid") for r in self.kg_context.relations if r.get("_rid")}

        for kw in keywords:
            entities = self.graph.search_entities(kw, top_k=5)
            for rank, entity in enumerate(entities):
                # Score by search rank
                score = 1.0 / (1.0 + rank)
                for cid in entity.source_chunk_ids:
                    chunk_scores[cid] = max(chunk_scores.get(cid, 0), score)

                # Collect entity description (dedup against local)
                if entity.entity_id not in _seen_eids and entity.description:
                    _seen_eids.add(entity.entity_id)
                    self.kg_context.entities.append(
                        {
                            "name": entity.name,
                            "type": entity.entity_type,
                            "description": entity.description,
                            "_eid": entity.entity_id,
                        }
                    )

                # Also collect relation chunks and descriptions
                relations = self.graph.get_relations(entity.entity_id)
                for rel in relations:
                    rel_score = score * 0.6 * rel.weight
                    for cid in rel.source_chunk_ids:
                        chunk_scores[cid] = max(chunk_scores.get(cid, 0), rel_score)

                    # Collect relation description (dedup against local + earlier global)
                    if rel.relation_id not in _seen_rids and rel.description:
                        _seen_rids.add(rel.relation_id)
                        src_ent = self.graph.get_entity(rel.source_entity)
                        tgt_ent = self.graph.get_entity(rel.target_entity)
                        self.kg_context.relations.append(
                            {
                                "source": src_ent.name if src_ent else rel.source_entity,
                                "target": tgt_ent.name if tgt_ent else rel.target_entity,
                                "keywords": rel.keywords,
                                "description": rel.description,
                                "_rid": rel.relation_id,
                            }
                        )

        return chunk_scores

    def _community_retrieval(self, query: str) -> dict[str, float]:
        """
        Community retrieval: embed query -> cosine search over
        community summary embeddings -> collect member chunk_ids.

        Also collects community summaries into ``self.kg_context``
        for synthesized context injection.
        """
        cw = getattr(self.cfg, "community_weight", 0.0)
        if cw <= 0 or self.embedder is None:
            return {}

        try:
            query_vec = self.embedder.embed_texts([query])[0]
        except Exception:
            return {}

        top_k = getattr(self.cfg, "community_top_k", 5)
        matches = self.graph.search_communities(query_vec, top_k=top_k)
        if not matches:
            return {}

        chunk_scores: dict[str, float] = {}
        for community, sim_score in matches:
            if sim_score < 0.3:  # skip very low similarity
                continue

            # Collect community summary for KG context
            if community.summary and community.summary != community.title:
                self.kg_context.community_summaries.append(
                    {
                        "title": community.title,
                        "summary": community.summary,
                    }
                )

            member_set = set(community.entity_ids)
            # Collect chunks from all community members
            for eid in community.entity_ids:
                entity = self.graph.get_entity(eid)
                if entity is None:
                    continue
                for cid in entity.source_chunk_ids:
                    chunk_scores[cid] = max(chunk_scores.get(cid, 0), sim_score)
                # Also include relation chunks within the community
                for rel in self.graph.get_relations(eid):
                    if rel.target_entity in member_set or rel.source_entity in member_set:
                        for cid in rel.source_chunk_ids:
                            chunk_scores[cid] = max(chunk_scores.get(cid, 0), sim_score * 0.7)

        return chunk_scores

    def _relation_retrieval(self, query: str) -> dict[str, float]:
        """
        Relation semantic search: embed query -> cosine search over
        relation description embeddings -> collect chunk_ids.

        Also collects relation descriptions into ``self.kg_context``
        (deduped against those already captured by local retrieval).
        """
        rw = getattr(self.cfg, "relation_weight", 0.0)
        if rw <= 0 or self.embedder is None:
            return {}

        try:
            query_vec = self.embedder.embed_texts([query])[0]
        except Exception:
            return {}

        top_k = getattr(self.cfg, "relation_top_k", 10)
        matches = self.graph.search_relations_by_embedding(query_vec, top_k=top_k)
        if not matches:
            return {}

        # Build set of already-collected relation IDs (from local retrieval)
        _existing_rids = {r.get("_rid") for r in self.kg_context.relations if r.get("_rid")}

        chunk_scores: dict[str, float] = {}
        for rel, sim_score in matches:
            if sim_score < 0.3:
                continue
            for cid in rel.source_chunk_ids:
                chunk_scores[cid] = max(chunk_scores.get(cid, 0), sim_score)

            # Collect relation description (dedup by relation_id)
            if rel.relation_id not in _existing_rids and rel.description:
                src_ent = self.graph.get_entity(rel.source_entity)
                tgt_ent = self.graph.get_entity(rel.target_entity)
                self.kg_context.relations.append(
                    {
                        "source": src_ent.name if src_ent else rel.source_entity,
                        "target": tgt_ent.name if tgt_ent else rel.target_entity,
                        "keywords": rel.keywords,
                        "description": rel.description,
                        "_rid": rel.relation_id,
                    }
                )
                _existing_rids.add(rel.relation_id)

        return chunk_scores

    def _merge_scores(
        self,
        local: dict[str, float],
        global_: dict[str, float],
        community: dict[str, float],
        relation: dict[str, float],
    ) -> dict[str, float]:
        """Weighted merge of all retrieval levels."""
        merged: dict[str, float] = {}
        lw = self.cfg.local_weight
        gw = self.cfg.global_weight
        cw = getattr(self.cfg, "community_weight", 0.0)
        rw = getattr(self.cfg, "relation_weight", 0.0)

        all_ids = set(local) | set(global_) | set(community) | set(relation)
        for cid in all_ids:
            score = (
                lw * local.get(cid, 0)
                + gw * global_.get(cid, 0)
                + cw * community.get(cid, 0)
                + rw * relation.get(cid, 0)
            )
            merged[cid] = score

        return merged

    def _verify_chunks(
        self,
        chunk_scores: dict[str, float],
    ) -> list[ScoredChunk]:
        """Verify chunk_ids exist in the relational store."""
        if not chunk_scores:
            return []

        verified = []
        # Batch check: get chunks by ID
        for cid, score in chunk_scores.items():
            chunk = self.rel.get_chunk(cid)
            if chunk is not None:
                verified.append(
                    ScoredChunk(
                        chunk_id=cid,
                        score=score,
                        source="kg",
                    )
                )

        return verified
