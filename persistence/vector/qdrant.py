"""
Qdrant VectorStore.

Connects to a Qdrant server (local or cloud) via qdrant-client.
Chunk metadata is stored as Qdrant payload fields for filtering.

Install: pip install qdrant-client
"""

from __future__ import annotations

import logging
from typing import Any

from config import QdrantConfig

from .base import VectorHit, VectorItem

log = logging.getLogger(__name__)

_DISTANCE_MAP = {
    "cosine": "Cosine",
    "l2": "Euclid",
    "ip": "Dot",
}


class QdrantStore:
    backend = "qdrant"

    def __init__(self, cfg: QdrantConfig):
        self.cfg = cfg
        self.dimension = cfg.dimension
        self._client = None

    # -------------------------------------------------------------------
    def connect(self) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise RuntimeError("QdrantStore requires qdrant-client: pip install qdrant-client") from e

        self._client = QdrantClient(
            url=self.cfg.url,
            api_key=self.cfg.api_key,
            prefer_grpc=self.cfg.prefer_grpc,
            timeout=self.cfg.timeout,
        )
        log.info("QdrantStore connected: %s", self.cfg.url)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # -------------------------------------------------------------------
    def ensure_schema(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self._client is None:
            raise RuntimeError("QdrantStore not connected")

        distance = getattr(Distance, _DISTANCE_MAP[self.cfg.distance])
        collections = [c.name for c in self._client.get_collections().collections]
        if self.cfg.collection_name not in collections:
            self._client.create_collection(
                collection_name=self.cfg.collection_name,
                vectors_config=VectorParams(
                    size=self.cfg.dimension,
                    distance=distance,
                ),
            )
            log.info(
                "Qdrant collection created: %s (dim=%d, distance=%s)",
                self.cfg.collection_name,
                self.cfg.dimension,
                self.cfg.distance,
            )
        else:
            log.info("Qdrant collection exists: %s", self.cfg.collection_name)

    # -------------------------------------------------------------------
    def upsert(self, items: list[VectorItem]) -> None:
        if not items:
            return
        from qdrant_client.models import PointStruct

        points = []
        for it in items:
            payload = dict(it.metadata)
            payload["doc_id"] = it.doc_id
            payload["parse_version"] = it.parse_version
            points.append(
                PointStruct(
                    id=it.chunk_id,
                    vector=it.embedding,
                    payload=payload,
                )
            )
        self._client.upsert(
            collection_name=self.cfg.collection_name,
            points=points,
        )

    # -------------------------------------------------------------------
    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        from qdrant_client.models import PointIdsList

        self._client.delete(
            collection_name=self.cfg.collection_name,
            points_selector=PointIdsList(points=chunk_ids),
        )

    def delete_parse_version(self, doc_id: str, parse_version: int) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self._client.delete(
            collection_name=self.cfg.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                    FieldCondition(key="parse_version", match=MatchValue(value=parse_version)),
                ]
            ),
        )

    # -------------------------------------------------------------------
    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        qfilter = None
        if filter:
            conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filter.items()]
            qfilter = Filter(must=conditions)

        results = self._client.search(
            collection_name=self.cfg.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qfilter,
        )
        hits: list[VectorHit] = []
        for r in results:
            payload = r.payload or {}
            hits.append(
                VectorHit(
                    chunk_id=str(r.id),
                    score=r.score,  # Qdrant returns similarity score directly
                    doc_id=payload.get("doc_id"),
                    parse_version=payload.get("parse_version"),
                    metadata=payload,
                )
            )
        return hits
