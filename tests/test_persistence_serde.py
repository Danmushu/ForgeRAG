"""Pure-Python tests for persistence.serde (no DB required)."""

from __future__ import annotations

from parser.schema import (
    BackendAttempt,
    Block,
    BlockType,
    Chunk,
    Complexity,
    DocFormat,
    DocProfile,
    DocTree,
    ParseTrace,
    TreeNode,
)
from persistence.serde import (
    block_to_row,
    chunk_to_row,
    profile_from_dict,
    profile_to_dict,
    row_to_block,
    row_to_chunk,
    trace_from_dict,
    trace_to_dict,
    tree_from_dict,
    tree_to_dict,
)


def _sample_profile() -> DocProfile:
    return DocProfile(
        page_count=10,
        format=DocFormat.PDF,
        file_size_bytes=1024,
        text_density=500.0,
        scanned_ratio=0.1,
        has_embedded_toc=True,
        has_multicolumn=False,
        table_density=0.3,
        figure_count=2,
        heading_hint_strength=0.8,
        complexity=Complexity.MEDIUM,
        needed_tier=1,
    )


def _sample_trace() -> ParseTrace:
    t = ParseTrace()
    t.attempts.append(
        BackendAttempt(
            backend="pymupdf",
            tier=0,
            started_at=1.0,
            duration_ms=100,
            quality_score=0.9,
            status="ok",
        )
    )
    t.final_backend = "pymupdf"
    t.final_tier = 0
    t.final_quality = 0.9
    t.total_duration_ms = 100
    return t


class TestProfileTrace:
    def test_profile_roundtrip(self):
        p = _sample_profile()
        d = profile_to_dict(p)
        p2 = profile_from_dict(d)
        assert p2 == p

    def test_trace_roundtrip(self):
        t = _sample_trace()
        d = trace_to_dict(t)
        t2 = trace_from_dict(d)
        assert t2.final_backend == "pymupdf"
        assert len(t2.attempts) == 1
        assert t2.attempts[0].status == "ok"


class TestBlock:
    def test_block_roundtrip(self):
        b = Block(
            block_id="doc:1:1:1",
            doc_id="doc",
            parse_version=1,
            page_no=1,
            seq=1,
            bbox=(10.0, 20.0, 30.0, 40.0),
            type=BlockType.HEADING,
            level=1,
            text="Title",
            confidence=0.95,
            cross_ref_targets=["doc:1:2:3"],
        )
        row = block_to_row(b)
        assert row["bbox_x0"] == 10.0
        assert row["type"] == "heading"
        assert row["cross_ref_targets"] == ["doc:1:2:3"]
        b2 = row_to_block(row)
        assert b2.bbox == b.bbox
        assert b2.type == BlockType.HEADING
        assert b2.cross_ref_targets == ["doc:1:2:3"]


class TestTree:
    def test_tree_roundtrip(self):
        root = TreeNode(
            node_id="doc:1:n1",
            doc_id="doc",
            parse_version=1,
            parent_id=None,
            level=0,
            title="root",
            page_start=1,
            page_end=5,
            children=["doc:1:n2"],
        )
        child = TreeNode(
            node_id="doc:1:n2",
            doc_id="doc",
            parse_version=1,
            parent_id="doc:1:n1",
            level=1,
            title="Intro",
            page_start=1,
            page_end=5,
            block_ids=["doc:1:1:1"],
            element_types=["paragraph"],
            table_count=1,
            figure_count=0,
            content_hash="abc123",
        )
        tree = DocTree(
            doc_id="doc",
            parse_version=1,
            root_id=root.node_id,
            nodes={root.node_id: root, child.node_id: child},
            quality_score=0.85,
            generation_method="toc",
        )
        d = tree_to_dict(tree)
        t2 = tree_from_dict(d)
        assert t2.root_id == tree.root_id
        assert t2.nodes[child.node_id].table_count == 1
        assert t2.nodes[child.node_id].content_hash == "abc123"
        assert t2.generation_method == "toc"


class TestChunk:
    def test_chunk_roundtrip(self):
        c = Chunk(
            chunk_id="doc:1:c1",
            doc_id="doc",
            parse_version=1,
            node_id="doc:1:n2",
            block_ids=["doc:1:1:1", "doc:1:1:2"],
            content="hello world",
            content_type="text",
            page_start=1,
            page_end=1,
            token_count=42,
            section_path=["root", "Intro"],
            ancestor_node_ids=["doc:1:n1"],
            cross_ref_chunk_ids=["doc:1:c5"],
        )
        row = chunk_to_row(c)
        assert row["block_ids"] == ["doc:1:1:1", "doc:1:1:2"]
        c2 = row_to_chunk(row)
        assert c2.section_path == ["root", "Intro"]
        assert c2.cross_ref_chunk_ids == ["doc:1:c5"]
