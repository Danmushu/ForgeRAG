"""Cache configuration."""

from pydantic import BaseModel


class CacheConfig(BaseModel):
    bm25_persistence: bool = True
    embedding_cache: bool = True
    bm25_path: str = "./storage/bm25_index.pkl"
    embedding_path: str = "./storage/embedding_cache.pkl"
