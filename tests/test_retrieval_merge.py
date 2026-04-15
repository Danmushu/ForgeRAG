"""Tests for retrieval.merge: RRF, sibling expansion, crossref expansion."""

from __future__ import annotations

from config import MergeConfig
from parser.schema import Chunk
from retrieval.merge import (
    expand_crossrefs,
    expand_descendants,
    expand_siblings,
    finalize_merged,
    rrf_merge,
)
from retrieval.types import MergedChunk, ScoredChunk

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _chunk(chunk_id, node_id="n1", content="body", cross_refs=None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d1",
        parse_version=1,
        node_id=node_id,
        block_ids=[f"b_{chunk_id}"],
        content=content,
        content_type="text",
        page_start=1,
        page_end=1,
        token_count=10,
        cross_ref_chunk_ids=list(cross_refs or []),
    )


class FakeStore:
    """Minimal relational store for merge tests."""

    backend = "fake"

    def __init__(self, chunks: list[Chunk]):
        self.by_id = {c.chunk_id: c for c in chunks}

    def _to_row(self, c: Chunk) -> dict:
        return {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "parse_version": c.parse_version,
            "node_id": c.node_id,
            "content": c.content,
            "content_type": c.content_type,
            "block_ids": list(c.block_ids),
            "page_start": c.page_start,
            "page_end": c.page_end,
            "token_count": c.token_count,
            "section_path": list(c.section_path),
            "ancestor_node_ids": list(c.ancestor_node_ids),
            "cross_ref_chunk_ids": list(c.cross_ref_chunk_ids),
        }

    def get_chunks_by_ids(self, chunk_ids):
        return [self._to_row(self.by_id[c]) for c in chunk_ids if c in self.by_id]

    def get_chunks_by_node_ids(self, node_ids):
        s = set(node_ids)
        return [self._to_row(c) for c in self.by_id.values() if c.node_id in s]


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------


class TestRRF:
    def test_basic_merge(self):
        a = [ScoredChunk("c1", 0.9, "vector"), ScoredChunk("c2", 0.8, "vector")]
        b = [ScoredChunk("c2", 0.7, "tree"), ScoredChunk("c3", 0.6, "tree")]
        merged = rrf_merge([a, b])
        assert set(merged.keys()) == {"c1", "c2", "c3"}
        # c2 appears in both lists -> highest fused score
        assert merged["c2"].rrf_score > merged["c1"].rrf_score
        assert merged["c2"].sources == {"vector", "tree"}

    def test_empty_inputs(self):
        assert rrf_merge([]) == {}
        assert rrf_merge([[], []]) == {}

    def test_rrf_constant_honored(self):
        a = [ScoredChunk("c1", 1.0, "vector")]
        merged_k60 = rrf_merge([a], k=60)
        merged_k10 = rrf_merge([a], k=10)
        # smaller k -> larger score for the same rank
        assert merged_k10["c1"].rrf_score > merged_k60["c1"].rrf_score


# ---------------------------------------------------------------------------
# Sibling expansion
# ---------------------------------------------------------------------------


class TestSiblingExpansion:
    def test_adds_small_node_siblings(self):
        chunks = [
            _chunk("c1", node_id="n1"),
            _chunk("c2", node_id="n1"),
            _chunk("c3", node_id="n1"),
        ]
        store = FakeStore(chunks)
        merged = {"c1": MergedChunk(chunk_id="c1", rrf_score=0.5)}
        expand_siblings(merged, store, MergeConfig())
        assert "c2" in merged and "c3" in merged
        assert merged["c2"].sources == {"expansion:sibling"}
        assert merged["c2"].rrf_score < 0.5

    def test_skips_large_nodes(self):
        chunks = [_chunk(f"c{i}", node_id="big") for i in range(10)]
        store = FakeStore(chunks)
        merged = {"c0": MergedChunk(chunk_id="c0", rrf_score=0.5)}
        expand_siblings(merged, store, MergeConfig(sibling_max_node_size=5))
        assert list(merged.keys()) == ["c0"]

    def test_respects_max_per_hit(self):
        chunks = [_chunk(f"c{i}", node_id="n1") for i in range(5)]
        store = FakeStore(chunks)
        merged = {"c0": MergedChunk(chunk_id="c0", rrf_score=0.5)}
        expand_siblings(
            merged,
            store,
            MergeConfig(sibling_max_node_size=5, sibling_max_per_hit=2),
        )
        added = [k for k in merged if k != "c0"]
        assert len(added) == 2

    def test_disabled(self):
        chunks = [_chunk("c1", node_id="n1"), _chunk("c2", node_id="n1")]
        store = FakeStore(chunks)
        merged = {"c1": MergedChunk(chunk_id="c1", rrf_score=0.5)}
        expand_siblings(merged, store, MergeConfig(sibling_expansion_enabled=False))
        assert list(merged.keys()) == ["c1"]


# ---------------------------------------------------------------------------
# Crossref expansion
# ---------------------------------------------------------------------------


class TestCrossrefExpansion:
    def test_follows_cross_ref_chunk_ids(self):
        chunks = [
            _chunk("c1", node_id="n1", cross_refs=["c2", "c3"]),
            _chunk("c2", node_id="n2", content="figure 1 content"),
            _chunk("c3", node_id="n3", content="table 1 content"),
        ]
        store = FakeStore(chunks)
        merged = {"c1": MergedChunk(chunk_id="c1", rrf_score=0.5, chunk=chunks[0])}
        expand_crossrefs(merged, store, MergeConfig())
        assert "c2" in merged and "c3" in merged
        assert merged["c2"].sources == {"expansion:crossref"}
        assert merged["c2"].parent_of == "c1"

    def test_max_per_hit(self):
        chunks = [
            _chunk("c1", cross_refs=[f"r{i}" for i in range(10)]),
            *[_chunk(f"r{i}") for i in range(10)],
        ]
        store = FakeStore(chunks)
        merged = {"c1": MergedChunk(chunk_id="c1", rrf_score=0.5, chunk=chunks[0])}
        expand_crossrefs(merged, store, MergeConfig(crossref_max_per_hit=3))
        added = [k for k in merged if k.startswith("r")]
        assert len(added) == 3

    def test_no_duplicate_if_already_in_merged(self):
        chunks = [
            _chunk("c1", cross_refs=["c2"]),
            _chunk("c2"),
        ]
        store = FakeStore(chunks)
        merged = {
            "c1": MergedChunk(chunk_id="c1", rrf_score=0.5, chunk=chunks[0]),
            "c2": MergedChunk(chunk_id="c2", rrf_score=0.3, chunk=chunks[1]),
        }
        expand_crossrefs(merged, store, MergeConfig())
        # c2 keeps its original rrf_score, not overwritten
        assert merged["c2"].rrf_score == 0.3


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Descendant expansion (PageIndex-style)
# ---------------------------------------------------------------------------


class FakeStoreWithTree(FakeStore):
    """Extends FakeStore with tree loading for descendant expansion."""

    def __init__(self, chunks, tree_json=None, doc_row=None):
        super().__init__(chunks)
        self._tree_json = tree_json
        self._doc_row = doc_row or {"active_parse_version": 1}

    def get_document(self, doc_id):
        return self._doc_row

    def load_tree(self, doc_id, parse_version):
        return self._tree_json


class TestDescendantExpansion:
    def test_thin_heading_expands_to_child_chunks(self):
        # heading chunk (thin: 3 tokens) in a parent node
        heading = _chunk("h1", node_id="parent", content="Introduction")
        heading_chunk_obj = heading
        # override token_count to be thin
        object.__setattr__(heading_chunk_obj, "token_count", 5)

        # child chunks in child nodes
        child1 = _chunk("c1", node_id="child_a", content="The method involves...")
        child2 = _chunk("c2", node_id="child_b", content="Experiments show that...")

        tree_json = {
            "nodes": {
                "root": {"children": ["parent"], "node_id": "root"},
                "parent": {"children": ["child_a", "child_b"], "node_id": "parent"},
                "child_a": {"children": [], "node_id": "child_a"},
                "child_b": {"children": [], "node_id": "child_b"},
            }
        }
        store = FakeStoreWithTree(
            [heading, child1, child2],
            tree_json=tree_json,
            doc_row={"active_parse_version": 1},
        )
        merged = {"h1": MergedChunk(chunk_id="h1", rrf_score=0.8, chunk=heading)}
        expand_descendants(merged, store, MergeConfig())
        # Both children should be pulled in
        assert "c1" in merged
        assert "c2" in merged
        assert merged["c1"].sources == {"expansion:descendant"}
        assert merged["c1"].parent_of == "h1"

    def test_thick_chunk_not_expanded(self):
        # chunk with enough tokens — not a heading, shouldn't expand
        body = _chunk("b1", node_id="parent", content="x " * 100)
        # Ensure token_count is above the descendant_min_token_threshold (80)
        object.__setattr__(body, "token_count", 100)
        tree_json = {
            "nodes": {
                "parent": {"children": ["child"], "node_id": "parent"},
                "child": {"children": [], "node_id": "child"},
            }
        }
        child = _chunk("c1", node_id="child")
        store = FakeStoreWithTree(
            [body, child],
            tree_json=tree_json,
            doc_row={"active_parse_version": 1},
        )
        merged = {"b1": MergedChunk(chunk_id="b1", rrf_score=0.5, chunk=body)}
        expand_descendants(merged, store, MergeConfig())
        # body has 100 tokens > threshold, should NOT expand
        assert "c1" not in merged

    def test_leaf_node_not_expanded(self):
        leaf = _chunk("l1", node_id="leaf", content="Short")
        tree_json = {
            "nodes": {
                "leaf": {"children": [], "node_id": "leaf"},
            }
        }
        store = FakeStoreWithTree(
            [leaf],
            tree_json=tree_json,
            doc_row={"active_parse_version": 1},
        )
        merged = {"l1": MergedChunk(chunk_id="l1", rrf_score=0.5, chunk=leaf)}
        expand_descendants(merged, store, MergeConfig())
        assert list(merged.keys()) == ["l1"]

    def test_respects_max_chunks(self):
        heading = _chunk("h1", node_id="parent", content="Title")
        children = [_chunk(f"c{i}", node_id=f"child_{i}") for i in range(20)]
        tree_json = {
            "nodes": {
                "parent": {
                    "children": [f"child_{i}" for i in range(20)],
                    "node_id": "parent",
                },
                **{f"child_{i}": {"children": [], "node_id": f"child_{i}"} for i in range(20)},
            }
        }
        store = FakeStoreWithTree(
            [heading, *children],
            tree_json=tree_json,
            doc_row={"active_parse_version": 1},
        )
        merged = {"h1": MergedChunk(chunk_id="h1", rrf_score=0.5, chunk=heading)}
        expand_descendants(merged, store, MergeConfig(descendant_max_chunks=3))
        added = [k for k in merged if k != "h1"]
        assert len(added) == 3

    def test_disabled(self):
        heading = _chunk("h1", node_id="parent", content="Title")
        child = _chunk("c1", node_id="child")
        tree_json = {
            "nodes": {
                "parent": {"children": ["child"], "node_id": "parent"},
                "child": {"children": [], "node_id": "child"},
            }
        }
        store = FakeStoreWithTree(
            [heading, child],
            tree_json=tree_json,
            doc_row={"active_parse_version": 1},
        )
        merged = {"h1": MergedChunk(chunk_id="h1", rrf_score=0.5, chunk=heading)}
        expand_descendants(merged, store, MergeConfig(descendant_expansion_enabled=False))
        assert list(merged.keys()) == ["h1"]


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------


class TestFinalize:
    def test_sort_and_cap(self):
        merged = {
            "c1": MergedChunk(chunk_id="c1", rrf_score=0.5, sources={"vector"}),
            "c2": MergedChunk(chunk_id="c2", rrf_score=0.9, sources={"vector"}),
            "c3": MergedChunk(chunk_id="c3", rrf_score=0.5, sources={"vector", "tree"}),
        }
        items = finalize_merged(
            merged,
            base_top_k=2,
            cfg=MergeConfig(global_budget_multiplier=1.5, candidate_limit=100),
        )
        # c2 first (highest score), then c3 (more sources), then c1
        assert items[0].chunk_id == "c2"
        assert items[1].chunk_id == "c3"
        assert len(items) == 3  # cap = max(candidate_limit, base*mult) = 3
