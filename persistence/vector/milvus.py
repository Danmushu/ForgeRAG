"""
Milvus VectorStore.

Connects to a Milvus server via pymilvus MilvusClient.
Uses a collection with chunk_id as primary key and metadata
fields for filtering.

Install: pip install pymilvus
"""

from __future__ import annotations

import logging
from typing import Any

from config import MilvusConfig

from .base import VectorHit, VectorItem

log = logging.getLogger(__name__)

_METRIC_MAP = {
    "cosine": "COSINE",
    "l2": "L2",
    "ip": "IP",
}


class MilvusStore:
    backend = "milvus"

    def __init__(self, cfg: MilvusConfig):
        self.cfg = cfg
        self.dimension = cfg.dimension
        self._client = None

    # -------------------------------------------------------------------
    def connect(self) -> None:
        try:
            from pymilvus import MilvusClient
        except ImportError as e:
            raise RuntimeError("MilvusStore requires pymilvus: pip install pymilvus") from e

        kwargs: dict[str, Any] = {"uri": self.cfg.uri}
        if self.cfg.token:
            kwargs["token"] = self.cfg.token
        self._client = MilvusClient(**kwargs)
        log.info("MilvusStore connected: %s", self.cfg.uri)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # -------------------------------------------------------------------
    def ensure_schema(self) -> None:
        from pymilvus import CollectionSchema, DataType, FieldSchema

        if self._client is None:
            raise RuntimeError("MilvusStore not connected")

        if self._client.has_collection(self.cfg.collection_name):
            log.info("Milvus collection exists: %s", self.cfg.collection_name)
            return

        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=512),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.cfg.dimension),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="parse_version", dtype=DataType.INT64),
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="page_start", dtype=DataType.INT64),
            FieldSchema(name="page_end", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
        metric = _METRIC_MAP[self.cfg.distance]

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=self.cfg.index_type,
            metric_type=metric,
            params={"M": 16, "efConstruction": 64} if self.cfg.index_type == "HNSW" else {},
        )

        self._client.create_collection(
            collection_name=self.cfg.collection_name,
            schema=schema,
            index_params=index_params,
        )
        log.info(
            "Milvus collection created: %s (dim=%d, metric=%s, index=%s)",
            self.cfg.collection_name,
            self.cfg.dimension,
            metric,
            self.cfg.index_type,
        )

    # -------------------------------------------------------------------
    def upsert(self, items: list[VectorItem]) -> None:
        if not items:
            return
        data = []
        for it in items:
            row = {
                "chunk_id": it.chunk_id,
                "embedding": it.embedding,
                "doc_id": it.doc_id,
                "parse_version": it.parse_version,
                "node_id": it.metadata.get("node_id", ""),
                "content_type": it.metadata.get("content_type", ""),
                "page_start": it.metadata.get("page_start", 0),
                "page_end": it.metadata.get("page_end", 0),
            }
            data.append(row)
        self._client.upsert(
            collection_name=self.cfg.collection_name,
            data=data,
        )

    # -------------------------------------------------------------------
    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        # Milvus delete by primary key expression
        ids_str = ", ".join(f'"{cid}"' for cid in chunk_ids)
        self._client.delete(
            collection_name=self.cfg.collection_name,
            filter=f"chunk_id in [{ids_str}]",
        )

    def delete_parse_version(self, doc_id: str, parse_version: int) -> None:
        self._client.delete(
            collection_name=self.cfg.collection_name,
            filter=f'doc_id == "{doc_id}" and parse_version == {parse_version}',
        )

    # -------------------------------------------------------------------
    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        expr = ""
        if filter:
            parts = []
            for k, v in filter.items():
                if isinstance(v, str):
                    parts.append(f'{k} == "{v}"')
                else:
                    parts.append(f"{k} == {v}")
            expr = " and ".join(parts)

        results = self._client.search(
            collection_name=self.cfg.collection_name,
            data=[query_vector],
            limit=top_k,
            filter=expr or None,
            output_fields=["doc_id", "parse_version", "node_id", "content_type", "page_start", "page_end"],
        )
        hits: list[VectorHit] = []
        for batch in results:
            for r in batch:
                entity = r.get("entity", {})
                hits.append(
                    VectorHit(
                        chunk_id=r["id"],
                        score=_distance_to_score(r["distance"], self.cfg.distance),
                        doc_id=entity.get("doc_id"),
                        parse_version=entity.get("parse_version"),
                        metadata=entity,
                    )
                )
        return hits


def _distance_to_score(distance: float, metric: str) -> float:
    if metric == "cosine":
        return 1.0 - distance
    if metric == "ip":
        return distance  # Milvus IP returns similarity directly
    return -distance  # L2: negate
