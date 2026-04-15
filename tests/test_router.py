"""Tests for parser.router -- uses fake backends, no real PDF needed."""

from __future__ import annotations

import pytest

from parser.backends.base import BackendUnavailable, ParserBackend
from parser.router import NoBackendAvailable, Router
from parser.schema import (
    Complexity,
    DocFormat,
    DocProfile,
    Page,
    ParsedDocument,
    ParseTrace,
)


def _profile(needed_tier: int, fmt: DocFormat = DocFormat.PDF) -> DocProfile:
    return DocProfile(
        page_count=10,
        format=fmt,
        file_size_bytes=1000,
        text_density=500,
        scanned_ratio=0.0,
        has_embedded_toc=False,
        has_multicolumn=False,
        table_density=0.0,
        figure_count=0,
        heading_hint_strength=0.5,
        complexity=Complexity.SIMPLE,
        needed_tier=needed_tier,
    )


def _empty_parsed(doc_id: str, fmt: DocFormat, profile: DocProfile) -> ParsedDocument:
    return ParsedDocument(
        doc_id=doc_id,
        filename="/tmp/x",
        format=fmt,
        parse_version=1,
        profile=profile,
        parse_trace=ParseTrace(),
        pages=[Page(page_no=1, width=595, height=842, block_ids=[])],
        blocks=[],
    )


# ---------------------------------------------------------------------------
# Fake backends
# ---------------------------------------------------------------------------


class FakeBackend(ParserBackend):
    def __init__(
        self,
        name: str,
        tier: int,
        *,
        supports: set[DocFormat] = {DocFormat.PDF},
        min_quality: float = 0.5,
        quality: float = 1.0,
        behavior: str = "ok",
    ):
        self.name = name
        self.tier = tier
        self.supports = supports
        self.min_quality = min_quality
        self._quality = quality
        self._behavior = behavior
        self.calls = 0
        self.blob_store = None  # type: ignore

    def available(self) -> bool:
        return self._behavior != "unavailable_at_start"

    def parse(self, path, doc_id, parse_version, profile):
        self.calls += 1
        if self._behavior == "unavailable":
            raise BackendUnavailable(f"{self.name} down")
        if self._behavior == "crash":
            raise RuntimeError(f"{self.name} boom")
        return _empty_parsed(doc_id, profile.format, profile)

    def self_check(self, result):
        return self._quality


# ---------------------------------------------------------------------------
# Chain building
# ---------------------------------------------------------------------------


class TestBuildChain:
    def test_prefers_tier_ge_needed_highest_first(self):
        b0 = FakeBackend("pymupdf", tier=0)
        b1 = FakeBackend("mineru", tier=1)
        b2 = FakeBackend("vlm", tier=2)
        r = Router([b0, b1, b2])
        chain = r.build_chain(DocFormat.PDF, needed_tier=1)
        # primary: tier>=1 high->low; then fallback tier<1; PyMuPDF kept last
        names = [b.name for b in chain]
        assert names.index("vlm") < names.index("mineru")
        assert names[-1] == "pymupdf"

    def test_pymupdf_always_last_for_pdf(self):
        b0 = FakeBackend("pymupdf", tier=0)
        b2 = FakeBackend("vlm", tier=2)
        r = Router([b0, b2])
        chain = r.build_chain(DocFormat.PDF, needed_tier=2)
        assert chain[-1].name == "pymupdf"

    def test_filters_by_format(self):
        b_pdf = FakeBackend("pymupdf", tier=0, supports={DocFormat.PDF})
        b_img = FakeBackend("ocrx", tier=1, supports={DocFormat.IMAGE})
        r = Router([b_pdf, b_img])
        chain = r.build_chain(DocFormat.IMAGE, needed_tier=1)
        assert [b.name for b in chain] == ["ocrx"]

    def test_empty_chain_when_no_support(self):
        b_pdf = FakeBackend("pymupdf", tier=0, supports={DocFormat.PDF})
        r = Router([b_pdf])
        chain = r.build_chain(DocFormat.DOCX, needed_tier=0)
        assert chain == []


# ---------------------------------------------------------------------------
# parse() fallback behavior
# ---------------------------------------------------------------------------


class TestParseFallback:
    def test_happy_path_picks_primary(self):
        primary = FakeBackend("mineru", tier=1, quality=0.9)
        backup = FakeBackend("pymupdf", tier=0, quality=0.5)
        r = Router([primary, backup])
        result = r.parse("x.pdf", "doc1", 1, _profile(needed_tier=1))
        assert result.parse_trace.final_backend == "mineru"
        assert primary.calls == 1
        assert backup.calls == 0

    def test_falls_through_on_quality_low(self):
        bad = FakeBackend("mineru", tier=1, min_quality=0.8, quality=0.3)
        good = FakeBackend("pymupdf", tier=0, min_quality=0.0, quality=0.5)
        r = Router([bad, good])
        result = r.parse("x.pdf", "doc1", 1, _profile(needed_tier=1))
        assert result.parse_trace.final_backend == "pymupdf"
        trace = result.parse_trace
        statuses = [a.status for a in trace.attempts]
        assert statuses == ["quality_low", "ok"]

    def test_falls_through_on_unavailable(self):
        gone = FakeBackend("mineru", tier=1, behavior="unavailable")
        good = FakeBackend("pymupdf", tier=0, quality=0.5, min_quality=0.0)
        r = Router([gone, good])
        result = r.parse("x.pdf", "doc1", 1, _profile(needed_tier=1))
        assert result.parse_trace.final_backend == "pymupdf"
        assert result.parse_trace.attempts[0].status == "unavailable"

    def test_falls_through_on_crash(self):
        crasher = FakeBackend("mineru", tier=1, behavior="crash")
        good = FakeBackend("pymupdf", tier=0, quality=0.5, min_quality=0.0)
        r = Router([crasher, good])
        result = r.parse("x.pdf", "doc1", 1, _profile(needed_tier=1))
        assert result.parse_trace.attempts[0].status == "error"
        assert result.parse_trace.final_backend == "pymupdf"

    def test_exhausted_chain_raises(self):
        b = FakeBackend("only", tier=0, behavior="crash")
        r = Router([b])
        with pytest.raises(NoBackendAvailable):
            r.parse("x.pdf", "doc1", 1, _profile(needed_tier=0))

    def test_empty_router_rejected(self):
        with pytest.raises(RuntimeError):
            Router([])
