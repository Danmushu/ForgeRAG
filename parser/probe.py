"""
Layer 0 document probe.

Cheap, zero-external-dependency inspection that answers three
questions about a document BEFORE any heavy backend is invoked:

    1. What format is it?
    2. Can PyMuPDF alone handle it, or do we need Layer 1/2?
    3. What's its complexity bucket?

The output (DocProfile) is persisted to documents.doc_profile_json
and is the sole input to router.route(). It does not depend on which
backends are actually installed -- it describes what the document
*wants*, and the router reconciles that with what's *available*.

All numeric thresholds come from ProbeConfig, not hardcoded.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from config import ProbeConfig

from .schema import Complexity, DocFormat, DocProfile

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


_EXT_TO_FORMAT = {
    ".pdf": DocFormat.PDF,
    ".docx": DocFormat.DOCX,
    ".doc": DocFormat.DOCX,
    ".pptx": DocFormat.PPTX,
    ".ppt": DocFormat.PPTX,
    ".xlsx": DocFormat.XLSX,
    ".xls": DocFormat.XLSX,
    ".html": DocFormat.HTML,
    ".htm": DocFormat.HTML,
    ".md": DocFormat.TEXT,
    ".txt": DocFormat.TEXT,
    ".png": DocFormat.IMAGE,
    ".jpg": DocFormat.IMAGE,
    ".jpeg": DocFormat.IMAGE,
    ".tif": DocFormat.IMAGE,
    ".tiff": DocFormat.IMAGE,
}


def detect_format(path: str | Path) -> DocFormat:
    ext = Path(path).suffix.lower()
    if ext not in _EXT_TO_FORMAT:
        raise ValueError(f"unsupported file extension: {ext}")
    return _EXT_TO_FORMAT[ext]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def probe(path: str | Path, cfg: ProbeConfig) -> DocProfile:
    """
    Inspect a document and return a DocProfile.

    For PDFs we open with PyMuPDF and compute real features. For other
    formats we return a minimal profile that routes to Docling (Layer 1
    equivalent for Office/HTML).
    """
    p = Path(path)
    fmt = detect_format(p)
    size = os.path.getsize(p)

    if fmt == DocFormat.PDF:
        return _probe_pdf(p, fmt, size, cfg)

    if fmt == DocFormat.IMAGE:
        # Single-image "document" -- always wants VLM or MinerU OCR.
        return DocProfile(
            page_count=1,
            format=fmt,
            file_size_bytes=size,
            text_density=0.0,
            scanned_ratio=1.0,
            has_embedded_toc=False,
            has_multicolumn=False,
            table_density=0.0,
            figure_count=1,
            heading_hint_strength=0.0,
            complexity=Complexity.COMPLEX,
            needed_tier=2,
        )

    # Office / HTML / Text: these formats are converted to PDF before
    # parsing, so the probe returns a minimal profile.
    return DocProfile(
        page_count=0,
        format=fmt,
        file_size_bytes=size,
        text_density=0.0,
        scanned_ratio=0.0,
        has_embedded_toc=False,
        has_multicolumn=False,
        table_density=0.0,
        figure_count=0,
        heading_hint_strength=0.0,
        complexity=Complexity.MEDIUM,
        needed_tier=1,
    )


# ---------------------------------------------------------------------------
# PDF-specific probing
# ---------------------------------------------------------------------------


def _probe_pdf(path: Path, fmt: DocFormat, size: int, cfg: ProbeConfig) -> DocProfile:
    import fitz  # PyMuPDF, lazy import

    doc = fitz.open(path)
    try:
        page_count = doc.page_count
        toc = doc.get_toc(simple=True) or []
        has_embedded_toc = len(toc) > 0

        total_chars = 0
        scanned_page_count = 0
        multicolumn_hits = 0
        figure_count = 0
        table_signal = 0.0

        # Heading heuristic: collect font sizes across the doc, then
        # see whether there is a clear top tier (prominent headings).
        font_sizes: Counter[int] = Counter()

        # We sample at most N pages for speed on huge PDFs.
        sample_pages = list(range(min(page_count, 50)))

        for i in sample_pages:
            page = doc.load_page(i)
            text_dict = page.get_text("dict")
            page_chars = 0
            x_centers: list[float] = []
            drawings_h, drawings_v = 0, 0

            # Count images
            images = page.get_images(full=False)
            figure_count += len(images)

            # Walk blocks
            for blk in text_dict.get("blocks", []):
                if blk.get("type", 0) == 1:
                    # image block -- contributes to scanned signal
                    continue
                lines = blk.get("lines", [])
                if not lines:
                    continue
                # bbox center x for multicolumn detection
                bb = blk.get("bbox")
                if bb:
                    x_centers.append((bb[0] + bb[2]) / 2.0)
                for line in lines:
                    for span in line.get("spans", []):
                        txt = span.get("text", "")
                        page_chars += len(txt)
                        sz = span.get("size")
                        if sz:
                            font_sizes[round(sz)] += len(txt)

            # Rough table signal: count horizontal/vertical path segments.
            # fitz exposes page.get_drawings() which is cheap.
            try:
                for d in page.get_drawings():
                    for item in d.get("items", []):
                        if not item or item[0] != "l":
                            continue
                        # item = ("l", p1, p2) line segment
                        p1, p2 = item[1], item[2]
                        if abs(p1.y - p2.y) < 0.5:
                            drawings_h += 1
                        elif abs(p1.x - p2.x) < 0.5:
                            drawings_v += 1
            except Exception:
                pass

            total_chars += page_chars

            # Scanned heuristic: page has images but almost no text.
            if images and page_chars < cfg.text_density_min:
                scanned_page_count += 1

            # Multicolumn heuristic: cluster block x-centers into 2+ groups
            if _has_multicolumn(x_centers, cfg.multicolumn_x_cluster_gap):
                multicolumn_hits += 1

            # Table signal: both H and V lines present in meaningful amounts
            if drawings_h >= 3 and drawings_v >= 3:
                table_signal += 1.0

        sampled = max(1, len(sample_pages))
        text_density = total_chars / sampled
        scanned_ratio = scanned_page_count / sampled
        has_multicolumn = multicolumn_hits / sampled > 0.3
        table_density = min(1.0, table_signal / sampled)
        heading_hint_strength = _heading_strength(font_sizes)

        complexity = _bucket_complexity(
            page_count=page_count,
            scanned_ratio=scanned_ratio,
            table_density=table_density,
            has_multicolumn=has_multicolumn,
            cfg=cfg,
        )
        needed_tier = _needed_tier(
            scanned_ratio=scanned_ratio,
            table_density=table_density,
            has_multicolumn=has_multicolumn,
            text_density=text_density,
            cfg=cfg,
        )

        return DocProfile(
            page_count=page_count,
            format=fmt,
            file_size_bytes=size,
            text_density=text_density,
            scanned_ratio=scanned_ratio,
            has_embedded_toc=has_embedded_toc,
            has_multicolumn=has_multicolumn,
            table_density=table_density,
            figure_count=figure_count,
            heading_hint_strength=heading_hint_strength,
            complexity=complexity,
            needed_tier=needed_tier,
        )
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


def _has_multicolumn(x_centers: list[float], gap: float) -> bool:
    """
    Simple multi-column detector: sort x centers, look for a large
    gap in the middle region of the page that separates two
    non-trivial clusters.
    """
    if len(x_centers) < 6:
        return False
    xs = sorted(x_centers)
    # Find the largest gap
    max_gap = 0.0
    split_idx = -1
    for i in range(1, len(xs)):
        g = xs[i] - xs[i - 1]
        if g > max_gap:
            max_gap = g
            split_idx = i
    if max_gap < gap:
        return False
    left = split_idx
    right = len(xs) - split_idx
    # Both sides need to have meaningful content
    return left >= 3 and right >= 3


def _heading_strength(font_sizes: Counter[int]) -> float:
    """
    Return a 0~1 score for how distinguishable heading-sized text is
    from body text. If the top bucket has a clearly larger font and
    non-trivial char count, headings are extractable by rule.
    """
    if not font_sizes:
        return 0.0
    sizes = sorted(font_sizes.items(), key=lambda kv: kv[0], reverse=True)
    total = sum(c for _, c in sizes)
    if total == 0:
        return 0.0
    # Body = the most common size
    body_size, _ = max(font_sizes.items(), key=lambda kv: kv[1])
    # Heading candidates = sizes strictly larger than body
    heading_chars = sum(c for s, c in font_sizes.items() if s > body_size)
    if heading_chars == 0:
        return 0.0
    ratio = heading_chars / total
    # Sweet spot: 1%~15% of chars are headings -> strong signal
    if 0.01 <= ratio <= 0.15:
        return min(1.0, ratio / 0.05)
    if ratio < 0.01:
        return ratio / 0.01 * 0.5
    return max(0.0, 1.0 - (ratio - 0.15) / 0.35)


def _bucket_complexity(
    *,
    page_count: int,
    scanned_ratio: float,
    table_density: float,
    has_multicolumn: bool,
    cfg: ProbeConfig,
) -> Complexity:
    if (
        scanned_ratio >= cfg.complex_scanned_ratio
        or page_count >= cfg.complex_page_count
        or (table_density >= cfg.table_density_threshold and has_multicolumn)
    ):
        return Complexity.COMPLEX
    if page_count >= cfg.medium_page_count or has_multicolumn or table_density >= cfg.table_density_threshold:
        return Complexity.MEDIUM
    return Complexity.SIMPLE


def _needed_tier(
    *,
    scanned_ratio: float,
    table_density: float,
    has_multicolumn: bool,
    text_density: float,
    cfg: ProbeConfig,
) -> int:
    # Tier 2: scanned or image-heavy -> wants VLM (or MinerU+OCR)
    if scanned_ratio >= cfg.scanned_ratio_threshold:
        return 2
    if text_density < cfg.text_density_min:
        return 2
    # Tier 1: tables / multicolumn -> wants MinerU layout analysis
    if table_density >= cfg.table_density_threshold or has_multicolumn:
        return 1
    # Tier 0: PyMuPDF is enough
    return 0
