"""
Parser layer data contract.

All parser backends (PyMuPDF / MinerU / VLM / Docling) produce a
ParsedDocument that conforms to this schema. Downstream modules
(tree builder, chunker, retriever, citation resolver) depend only
on this file.

Coordinate system
-----------------
All bbox values use the native PDF coordinate system:
    - origin at bottom-left of the page
    - units in points (1 pt = 1/72 inch)
    - tuple order: (x0, y0, x1, y1) where x0<x1, y0<y1
Do NOT normalize. The frontend (PDF.js) handles conversion to
viewport coordinates via page.getViewport().convertToViewportRectangle.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

BBox = tuple[float, float, float, float]  # (x0, y0, x1, y1) in PDF points


class DocFormat(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    HTML = "html"
    TEXT = "text"
    IMAGE = "image"


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    FORMULA = "formula"
    CAPTION = "caption"
    HEADER = "header"  # page header (usually excluded from reading flow)
    FOOTER = "footer"  # page footer (usually excluded from reading flow)


class Complexity(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


# ---------------------------------------------------------------------------
# Layer 0 probe output
# ---------------------------------------------------------------------------


@dataclass
class DocProfile:
    """Cheap, always-computed features from the Layer-0 probe."""

    page_count: int
    format: DocFormat
    file_size_bytes: int

    text_density: float  # avg chars per page
    scanned_ratio: float  # fraction of pages dominated by images
    has_embedded_toc: bool
    has_multicolumn: bool
    table_density: float  # heuristic 0~1
    figure_count: int
    heading_hint_strength: float  # 0~1 based on font-size distribution

    complexity: Complexity
    needed_tier: int  # 0 / 1 / 2 -- the tier this doc *wants*


# ---------------------------------------------------------------------------
# Parse trace (observability + rerun decisions)
# ---------------------------------------------------------------------------


@dataclass
class BackendAttempt:
    backend: str
    tier: int
    started_at: float  # unix timestamp
    duration_ms: int
    quality_score: float  # 0~1, backend self_check
    status: Literal["ok", "quality_low", "error", "unavailable"]
    error_message: str | None = None


@dataclass
class ParseTrace:
    attempts: list[BackendAttempt] = field(default_factory=list)
    final_backend: str | None = None
    final_tier: int | None = None
    final_quality: float | None = None
    total_duration_ms: int = 0

    def record(self, attempt: BackendAttempt) -> None:
        self.attempts.append(attempt)
        self.total_duration_ms += attempt.duration_ms
        if attempt.status == "ok":
            self.final_backend = attempt.backend
            self.final_tier = attempt.tier
            self.final_quality = attempt.quality_score


# ---------------------------------------------------------------------------
# TOC (preserved verbatim if the source file has an embedded one)
# ---------------------------------------------------------------------------


@dataclass
class TocEntry:
    level: int  # 1-based
    title: str
    page_no: int  # 1-based page number
    children: list[TocEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """
    Smallest addressable unit of content. Every citation in the system
    ultimately resolves to one or more Block ids.

    block_id format: "{doc_id}:{parse_version}:{page_no}:{seq}"
    This makes ids stable within a parse_version and trivially
    decomposable for debugging.
    """

    block_id: str
    doc_id: str
    parse_version: int
    page_no: int  # 1-based
    seq: int  # order within the page
    bbox: BBox  # PDF points, origin bottom-left
    type: BlockType

    text: str  # for tables: markdown rendering
    level: int | None = None  # heading level, 1~6
    confidence: float = 1.0  # parser confidence 0~1

    # Table payload
    table_html: str | None = None
    table_markdown: str | None = None

    # Figure payload (stored via BlobStore; these are the lookup keys)
    figure_storage_key: str | None = None
    figure_mime: str | None = None
    figure_caption: str | None = None

    # Formula payload
    formula_latex: str | None = None

    # Normalizer flags -- never delete blocks, only mark them
    excluded: bool = False  # True for header/footer/noise
    excluded_reason: str | None = None

    # Cross-refs discovered at parse time (optional; filled by normalizer)
    caption_of: str | None = None  # this caption describes block_id X

    # Inline references this block makes to other blocks in the same doc,
    # e.g. text "as shown in Figure 3" -> [block_id of Figure 3].
    # Populated by normalizer._resolve_inline_references.
    cross_ref_targets: list[str] = field(default_factory=list)


@dataclass
class Page:
    page_no: int  # 1-based
    width: float  # in points
    height: float  # in points
    block_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level ParsedDocument -- the single contract with downstream
# ---------------------------------------------------------------------------


@dataclass
class ParsedDocument:
    doc_id: str
    filename: str
    format: DocFormat
    parse_version: int  # bumped on every re-parse

    profile: DocProfile
    parse_trace: ParseTrace

    pages: list[Page]
    blocks: list[Block]  # flat, reading order across pages
    toc: list[TocEntry] | None = None

    def reading_blocks(self) -> list[Block]:
        """Blocks in reading order, excluding headers/footers/noise."""
        return [b for b in self.blocks if not b.excluded]

    def blocks_by_id(self) -> dict[str, Block]:
        return {b.block_id: b for b in self.blocks}


# ---------------------------------------------------------------------------
# Citation (retrieval -> viewer highlight)
# ---------------------------------------------------------------------------


@dataclass
class HighlightRect:
    page_no: int
    bbox: BBox


# ---------------------------------------------------------------------------
# Document tree (PageIndex-style hierarchical index)
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """
    A single node in the document tree.

    A node represents either:
        - the whole document (root, level=0, title = filename)
        - a section/subsection identified from TOC, heading blocks,
          or LLM-inferred structure.

    block_ids holds only the blocks DIRECTLY owned by this node, not
    the union with descendants. Callers that need "all blocks under
    this subtree" should traverse children.

    node_id format: "{doc_id}:{parse_version}:n{seq}" where seq is
    the preorder index assigned at build time.
    """

    node_id: str
    doc_id: str
    parse_version: int
    parent_id: str | None  # None for root
    level: int  # 0 = root, 1.. = section depth
    title: str
    page_start: int  # 1-based, inclusive
    page_end: int  # 1-based, inclusive
    block_ids: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)  # ordered node_ids

    # Enrichment (cheap, computed at build time)
    element_types: list[str] = field(default_factory=list)  # BlockType values
    table_count: int = 0
    figure_count: int = 0
    content_hash: str = ""  # hash of concatenated block text

    # Deferred enrichment (filled by later passes)
    summary: str | None = None
    key_entities: list[str] = field(default_factory=list)
    cross_reference_targets: list[str] = field(default_factory=list)  # node_ids


@dataclass
class DocTree:
    """Flat storage for a document tree. Access via node_id lookups."""

    doc_id: str
    parse_version: int
    root_id: str
    nodes: dict[str, TreeNode]
    quality_score: float
    generation_method: Literal["toc", "headings", "llm", "page_groups", "fallback"]

    def get(self, node_id: str) -> TreeNode:
        return self.nodes[node_id]

    def root(self) -> TreeNode:
        return self.nodes[self.root_id]

    def leaves(self) -> list[TreeNode]:
        return [n for n in self.nodes.values() if not n.children]

    def walk_preorder(self, start: str | None = None) -> Iterator[TreeNode]:
        stack = [start or self.root_id]
        while stack:
            nid = stack.pop()
            node = self.nodes[nid]
            yield node
            # Push children in reverse so they pop in order
            for cid in reversed(node.children):
                stack.append(cid)

    def ancestors(self, node_id: str) -> list[TreeNode]:
        """Return ancestors from root to direct parent (excludes self)."""
        chain: list[TreeNode] = []
        node = self.nodes[node_id]
        while node.parent_id is not None:
            parent = self.nodes[node.parent_id]
            chain.append(parent)
            node = parent
        chain.reverse()
        return chain


# ---------------------------------------------------------------------------
# Chunks (retrieval unit)
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """
    A retrieval-unit chunk produced by the chunker.

    Invariants:
        - block_ids are contiguous within a single tree node (one
          chunk never spans multiple nodes).
        - content_type reflects the structural kind of content:
            "text":    ordinary prose / lists / headings
            "table":   a single table block
            "figure":  a single figure block
            "formula": a single formula block
            "mixed":   text chunk that happens to include a non-text
                       block because isolate_* was disabled for it

    chunk_id format: "{doc_id}:{parse_version}:c{seq}"
    """

    chunk_id: str
    doc_id: str
    parse_version: int
    node_id: str  # owning tree node (leaf or inner)
    block_ids: list[str]  # ordered, contiguous
    content: str  # joined block texts

    content_type: Literal["text", "table", "figure", "formula", "mixed"]
    page_start: int
    page_end: int
    token_count: int  # approximate, per ChunkerConfig.tokenizer

    # Structural context -- filled at chunking time from the tree.
    # section_path goes root -> owning node inclusive, by title.
    # ancestor_node_ids goes root -> direct parent (excludes owning node).
    section_path: list[str] = field(default_factory=list)
    ancestor_node_ids: list[str] = field(default_factory=list)

    # Sort key: negative y1 of the first block on page_start.
    # PDF y origin is bottom-left (higher y = higher on page), so
    # negating gives ascending sort = top-to-bottom reading order.
    y_sort: float = 0.0

    # Cross-references: other chunks that blocks in this chunk point at
    # via their block.cross_ref_targets. Filled in a second pass after
    # all chunks are emitted. Deduped, excludes self.
    cross_ref_chunk_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Citation (retrieval -> viewer highlight)
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    """
    A single answer-span citation. Produced by the retrieval/rerank
    layer and consumed by the frontend PDF viewer.

    The viewer reads `highlights` and draws annotation-layer rectangles
    on top of the PDF.js page; `page_no` is used for initial scroll.
    `file_id` is the FileStore identifier the viewer should use to
    fetch the source PDF blob.
    """

    citation_id: str  # short id, e.g. "c_12"
    chunk_id: str  # full chunk_id for traceability
    doc_id: str
    parse_version: int  # for version-mismatch detection
    block_ids: list[str]  # ordered, may span multiple blocks
    page_no: int  # first block's page, for jump
    highlights: list[HighlightRect]
    snippet: str  # <=200 chars, for hover preview
    score: float  # rerank score
    file_id: str | None = None  # FileStore file_id for PDF viewing (may be converted)
    source_file_id: str | None = None  # original file_id (only if converted, for download)
    source_format: str = ""  # original format, e.g. "docx" (empty if native PDF)
    open_url: str | None = None  # e.g. /viewer/{doc_id}?page=14&hl=c_12
