"""Tests for parser.probe -- uses the sample_pdf fixture."""

from __future__ import annotations

import pytest

from config import ProbeConfig
from parser.probe import detect_format, probe
from parser.schema import Complexity, DocFormat

pytest.importorskip("fitz")


class TestDetectFormat:
    def test_pdf(self, tmp_path):
        assert detect_format(tmp_path / "x.pdf") == DocFormat.PDF

    def test_image(self, tmp_path):
        assert detect_format(tmp_path / "x.png") == DocFormat.IMAGE

    def test_unsupported_raises(self, tmp_path):
        with pytest.raises(ValueError):
            detect_format(tmp_path / "x.zip")


class TestPDFProbe:
    def test_profile_basics(self, sample_pdf):
        cfg = ProbeConfig()
        profile = probe(sample_pdf, cfg)
        assert profile.format == DocFormat.PDF
        assert profile.page_count == 4
        assert profile.has_embedded_toc is True
        assert profile.file_size_bytes > 0
        assert profile.text_density > 0

    def test_simple_doc_routes_to_tier_0(self, sample_pdf):
        cfg = ProbeConfig()
        profile = probe(sample_pdf, cfg)
        assert profile.needed_tier == 0
        assert profile.complexity == Complexity.SIMPLE

    def test_heading_hint_is_positive(self, sample_pdf):
        # Page 1 has a 22pt title vs 11pt body -> strong signal
        profile = probe(sample_pdf, ProbeConfig())
        assert profile.heading_hint_strength > 0

    def test_not_marked_as_scanned(self, sample_pdf):
        profile = probe(sample_pdf, ProbeConfig())
        assert profile.scanned_ratio == 0.0


class TestOfficeAndImage:
    def test_image_file_routes_to_tier_2(self, tmp_path):
        p = tmp_path / "blank.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")  # fake magic is fine; probe only reads ext
        profile = probe(p, ProbeConfig())
        assert profile.needed_tier == 2
        assert profile.complexity == Complexity.COMPLEX

    def test_docx_routes_to_tier_1_without_opening(self, tmp_path):
        p = tmp_path / "doc.docx"
        p.write_bytes(b"fake")
        profile = probe(p, ProbeConfig())
        assert profile.format == DocFormat.DOCX
        assert profile.needed_tier == 1
