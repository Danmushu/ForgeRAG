"""Tests for retrieval.bm25.InMemoryBM25Index."""

from __future__ import annotations

from config import BM25Config
from retrieval.bm25 import InMemoryBM25Index, tokenize


class TestTokenize:
    def test_english_lowercase(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_cjk_chars_split(self):
        # 4 CJK chars -> 4 tokens
        assert tokenize("你好世界") == ["你", "好", "世", "界"]

    def test_mixed(self):
        assert tokenize("BGE模型 is great") == ["bge", "模", "型", "is", "great"]

    def test_empty(self):
        assert tokenize("") == []
        assert tokenize("   ") == []


class TestBM25Search:
    def _index(self) -> InMemoryBM25Index:
        idx = InMemoryBM25Index(BM25Config())
        idx.add("c1", "d1", "pytorch deep learning tutorial")
        idx.add("c2", "d1", "loss function and gradient descent")
        idx.add("c3", "d2", "transformers attention mechanism")
        idx.add("c4", "d2", "BERT language model pretraining")
        idx.add("c5", "d3", "完全无关的中文文档")
        idx.finalize()
        return idx

    def test_chunk_search_ranks_exact_match_top(self):
        idx = self._index()
        results = idx.search_chunks("gradient descent", top_k=3)
        assert len(results) >= 1
        assert results[0][0] == "c2"

    def test_doc_search_groups_by_doc(self):
        idx = self._index()
        results = idx.search_docs("attention transformers", top_k=3)
        top = [d for d, _ in results]
        assert top[0] == "d2"

    def test_no_match_returns_empty(self):
        idx = self._index()
        results = idx.search_chunks("quantum gravity", top_k=5)
        assert results == []

    def test_empty_query_returns_empty(self):
        idx = self._index()
        assert idx.search_chunks("", top_k=5) == []

    def test_cjk_search(self):
        idx = self._index()
        results = idx.search_chunks("中文", top_k=3)
        # At least matches c5 via "中" and "文"
        assert any(cid == "c5" for cid, _ in results)

    def test_finalize_is_idempotent(self):
        idx = self._index()
        idx.finalize()
        idx.finalize()
        assert len(idx) == 5
