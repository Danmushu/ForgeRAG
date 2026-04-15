"""
Weaviate VectorStore (v4 client).

Connects to a Weaviate instance via weaviate-client v4 API.
Uses a collection with chunk_id stored as a property and
metadata fields for filtering.

Install: pip install weaviate-client
"""

from __future__ import annotations

import logging
from typing import Any

from config import WeaviateConfig

from .base import VectorHit, VectorItem

log = logging.getLogger(__name__)

_DISTANCE_MAP = {
    "cosine": "cosine",
    "l2": "l2-squared",
    "dot": "dot",
}

# Metadata properties stored on each object
_PAYLOAD_PROPS = ["chunk_id", "doc_id", "parse_version", "node_id", "content_type", "page_start", "page_end"]


class WeaviateStore:
    backend = "weaviate"

    def __init__(self, cfg: WeaviateConfig):
        self.cfg = cfg
        self.dimension = cfg.dimension
        self._client = None
        self._collection = None

    # -------------------------------------------------------------------
    def connect(self) -> None:
        try:
            import weaviate
        except ImportError as e:
            raise RuntimeError("WeaviateStore requires weaviate-client: pip install weaviate-client") from e

        if self.cfg.api_key:
            auth = weaviate.auth.AuthApiKey(self.cfg.api_key)
            self._client = weaviate.connect_to_custom(
                http_host=self._parse_host(self.cfg.url),
                http_port=self._parse_port(self.cfg.url),
                http_secure=self.cfg.url.startswith("https"),
                grpc_host=self._parse_host(self.cfg.url),
                grpc_port=50051,
                grpc_secure=self.cfg.url.startswith("https"),
                auth_credentials=auth,
            )
        else:
            self._client = weaviate.connect_to_local(
                host=self._parse_host(self.cfg.url),
                port=self._parse_port(self.cfg.url),
            )
        log.info("WeaviateStore connected: %s", self.cfg.url)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._collection = None

    @staticmethod
    def _parse_host(url: str) -> str:
        from urllib.parse import urlparse

        return urlparse(url).hostname or "localhost"

    @staticmethod
    def _parse_port(url: str) -> int:
        from urllib.parse import urlparse

        port = urlparse(url).port
        return port or 8080

    # -------------------------------------------------------------------
    def ensure_schema(self) -> None:
        import weaviate.classes.config as wvc

        if self._client is None:
            raise RuntimeError("WeaviateStore not connected")

        distance_map = {
            "cosine": wvc.VectorDistances.COSINE,
            "l2": wvc.VectorDistances.L2_SQUARED,
            "dot": wvc.VectorDistances.DOT,
        }

        name = self.cfg.collection_name
        if self._client.collections.exists(name):
            self._collection = self._client.collections.get(name)
            log.info("Weaviate collection exists: %s", name)
        else:
            self._collection = self._client.collections.create(
                name=name,
                vectorizer_config=wvc.Configure.Vectorizer.none(),
                vector_index_config=wvc.Configure.VectorIndex.hnsw(
                    distance_metric=distance_map.get(self.cfg.distance, wvc.VectorDistances.COSINE),
                ),
                properties=[
                    wvc.Property(name="chunk_id", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="doc_id", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="parse_version", data_type=wvc.DataType.INT),
                    wvc.Property(name="node_id", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="content_type", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="page_start", data_type=wvc.DataType.INT),
                    wvc.Property(name="page_end", data_type=wvc.DataType.INT),
                ],
            )
            log.info("Weaviate collection created: %s", name)

    def _ensure_collection(self):
        if self._collection is None:
            self.ensure_schema()
        return self._collection

    # -------------------------------------------------------------------
    def upsert(self, items: list[VectorItem]) -> None:
        if not items:
            return
        import weaviate.classes.data as wvd

        col = self._ensure_collection()
        with col.batch.dynamic() as batch:
            for it in items:
                props = {
                    "chunk_id": it.chunk_id,
                    "doc_id": it.doc_id,
                    "parse_version": it.parse_version,
                    "node_id": it.metadata.get("node_id", ""),
                    "content_type": it.metadata.get("content_type", ""),
                    "page_start": it.metadata.get("page_start", 0),
                    "page_end": it.metadata.get("page_end", 0),
                }
                batch.add_object(
                    properties=props,
                    vector=it.embedding,
                    uuid=wvd.generate_uuid5(it.chunk_id),
                )

    # -------------------------------------------------------------------
    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        import weaviate.classes.data as wvd

        col = self._ensure_collection()
        for cid in chunk_ids:
            col.data.delete_by_id(wvd.generate_uuid5(cid))

    def delete_parse_version(self, doc_id: str, parse_version: int) -> None:
        import weaviate.classes.query as wvq

        col = self._ensure_collection()
        col.data.delete_many(
            where=wvq.Filter.by_property("doc_id").equal(doc_id)
            & wvq.Filter.by_property("parse_version").equal(parse_version)
        )

    # -------------------------------------------------------------------
    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        import weaviate.classes.query as wvq

        col = self._ensure_collection()
        wfilter = None
        if filter:
            conditions = []
            for k, v in filter.items():
                conditions.append(wvq.Filter.by_property(k).equal(v))
            if len(conditions) == 1:
                wfilter = conditions[0]
            else:
                wfilter = conditions[0]
                for c in conditions[1:]:
                    wfilter = wfilter & c

        result = col.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            filters=wfilter,
            return_metadata=wvq.MetadataQuery(distance=True),
        )
        hits: list[VectorHit] = []
        for obj in result.objects:
            props = obj.properties or {}
            dist = obj.metadata.distance if obj.metadata else 0.0
            hits.append(
                VectorHit(
                    chunk_id=props.get("chunk_id", str(obj.uuid)),
                    score=_distance_to_score(dist, self.cfg.distance),
                    doc_id=props.get("doc_id"),
                    parse_version=props.get("parse_version"),
                    metadata=props,
                )
            )
        return hits


def _distance_to_score(distance: float, metric: str) -> float:
    if metric == "cosine":
        return 1.0 - distance
    if metric == "dot":
        return -distance
    return -distance  # l2
