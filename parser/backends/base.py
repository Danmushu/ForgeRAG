"""
Parser backend abstract base class.

Every concrete backend (PyMuPDF, MinerU, VLM, Docling) implements
ParserBackend. The router iterates a list of backends and picks
the first one whose available() is True and whose self_check()
passes its configured min_quality.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..blob_store import BlobStore
from ..schema import DocFormat, DocProfile, ParsedDocument


class BackendUnavailable(RuntimeError):
    """Raised when a backend's dependencies / model / hardware are missing."""


class ParserBackend(ABC):
    #: short name used in ParseTrace and logs, e.g. "pymupdf"
    name: str
    #: 0 / 1 / 2, matching DocProfile.needed_tier
    tier: int
    #: set of DocFormat values the backend can parse
    supports: set[DocFormat]
    #: quality score below which the router should fall through
    min_quality: float

    def __init__(self, blob_store: BlobStore):
        self.blob_store = blob_store

    @abstractmethod
    def available(self) -> bool:
        """
        Return True iff this backend can actually run right now.
        Cheap check only -- no model loading. The router calls this
        once at startup to build the dispatch chain.
        """

    @abstractmethod
    def parse(self, path: str, doc_id: str, parse_version: int, profile: DocProfile) -> ParsedDocument:
        """
        Parse `path` into a ParsedDocument. Must populate doc_id,
        parse_version, profile, blocks, and pages. parse_trace is
        managed by the router, not by the backend.

        Raise BackendUnavailable if preconditions fail at call time
        (e.g. GPU OOM, model download failed). Any other exception
        is logged by the router and recorded as "error".
        """

    @abstractmethod
    def self_check(self, result: ParsedDocument) -> float:
        """
        Return a 0~1 quality score for the parse. The router compares
        this to self.min_quality to decide whether to accept or fall
        through to the next backend.
        """

    def supports_format(self, fmt: DocFormat) -> bool:
        return fmt in self.supports
