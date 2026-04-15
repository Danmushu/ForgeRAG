"""
Markdown heading reclassification for converted text/markdown files.

When a ``.txt`` or ``.md`` file is converted to PDF and then parsed
by PyMuPDF, Markdown headings (``# title``, ``## subtitle``, etc.)
lose their semantic meaning — they become plain PARAGRAPH blocks.

This module provides a post-parse pass that scans block text for
Markdown heading patterns and reclassifies matching blocks as
``BlockType.HEADING`` with the correct ``level``.  This lets the
tree builder use the heading-based strategy to produce a real
document tree instead of a flat fallback.

Additionally, it bumps ``DocProfile.heading_hint_strength`` so
the tree builder's strategy selector picks the heading path.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parser.schema import ParsedDocument

log = logging.getLogger(__name__)

# Patterns that indicate a Markdown heading.
# After PDF conversion + PyMuPDF parsing, leading ``#`` may survive
# in the block text, or the heading may appear as a bold-only line
# (``**Title**``).
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#+)?\s*$")

# Bold-only line that looks like a section title.
# Must be the ENTIRE block text (not just a bold phrase in a paragraph).
_MD_BOLD_HEADING_RE = re.compile(r"^\*{2,3}([^*\n]{3,120})\*{2,3}$")


def reclassify_md_headings(doc: ParsedDocument) -> int:
    """
    Scan *doc.blocks* in-place and reclassify Markdown headings.

    Returns the number of blocks reclassified.
    """
    from parser.schema import BlockType

    count = 0

    for block in doc.blocks:
        if block.excluded:
            continue
        # Only reclassify paragraphs (headings already classified are fine)
        if block.type != BlockType.PARAGRAPH:
            continue
        text = block.text.strip()
        if not text:
            continue

        # --- ``# heading`` through ``###### heading`` ---
        m = _MD_HEADING_RE.match(text)
        if m:
            level = len(m.group(1))
            # Clean title: remove leading #'s and inline bold markers
            clean = m.group(2).strip()
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", clean)
            block.type = BlockType.HEADING
            block.level = level
            block.text = clean
            count += 1
            continue

        # --- ``**bold-only line**`` as implicit heading ---
        # Only short, single-line bold text qualifies (avoids
        # reclassifying bold paragraphs).
        bm = _MD_BOLD_HEADING_RE.match(text)
        if bm and "\n" not in text:
            title = bm.group(1).strip()
            block.type = BlockType.HEADING
            # Bold-only headings don't declare a level; assign 2
            # (section-level) as a reasonable default.
            block.level = 2
            block.text = title
            count += 1
            continue

    if count:
        # Bump heading_hint_strength so the tree builder picks the
        # heading strategy instead of fallback.
        total_reading = sum(1 for b in doc.blocks if not b.excluded)
        if total_reading > 0:
            heading_ratio = count / total_reading
            # Use a generous strength: even a few headings are enough.
            doc.profile.heading_hint_strength = max(
                doc.profile.heading_hint_strength,
                min(heading_ratio * 5, 1.0),
            )
        log.info(
            "md_headings: reclassified %d blocks as headings (hint_strength=%.2f)",
            count,
            doc.profile.heading_hint_strength,
        )

    return count
