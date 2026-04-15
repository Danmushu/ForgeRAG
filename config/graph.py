"""Graph database configuration for Knowledge Graph storage."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class NetworkXConfig(BaseModel):
    path: str = "./storage/kg.json"


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    password_env: str | None = "NEO4J_PASSWORD"
    database: str = "neo4j"


class EntityDisambiguationConfig(BaseModel):
    """Embedding-based entity deduplication at upsert time."""

    enabled: bool = False
    similarity_threshold: float = 0.85
    candidate_top_k: int = 10


class CommunityDetectionConfig(BaseModel):
    """Leiden clustering + LLM community summaries."""

    enabled: bool = False
    resolution: float = 1.0
    min_community_size: int = 3
    # LLM for generating summaries
    provider_id: str | None = None
    model: str = "openai/gpt-4o-mini"
    api_key: str | None = None
    api_key_env: str | None = None
    api_base: str | None = None
    max_workers: int = 5
    timeout: float = 60.0


class GraphConfig(BaseModel):
    backend: Literal["networkx", "neo4j"] = "networkx"
    networkx: NetworkXConfig = Field(default_factory=NetworkXConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    entity_disambiguation: EntityDisambiguationConfig = Field(default_factory=EntityDisambiguationConfig)
    community_detection: CommunityDetectionConfig = Field(default_factory=CommunityDetectionConfig)
