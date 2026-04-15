"""Retrieval layer: dual-path + RRF merge + rerank + citations."""

from .bm25 import InMemoryBM25Index
from .citations import build_citations
from .merge import expand_crossrefs, expand_siblings, rrf_merge
from .pipeline import RetrievalPipeline
from .rerank import PassthroughReranker, Reranker, make_reranker
from .types import MergedChunk, RetrievalResult, ScoredChunk

__all__ = [
    "InMemoryBM25Index",
    "MergedChunk",
    "PassthroughReranker",
    "Reranker",
    "RetrievalPipeline",
    "RetrievalResult",
    "ScoredChunk",
    "build_citations",
    "expand_crossrefs",
    "expand_siblings",
    "make_reranker",
    "rrf_merge",
]
