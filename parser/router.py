"""
Backend router & fallback chain.

Given a DocProfile and a set of configured/available backends,
build an ordered list of candidates and try them in sequence.
Records every attempt into ParseTrace for observability.

Fallback rule
-------------
1. Filter by format support.
2. Sort by tier DESC, preferring tiers >= profile.needed_tier first,
   then lower tiers as fallback.
3. PyMuPDF (tier 0) is always appended last as the final safety net
   if it supports the format, even if its min_quality threshold is 0.
4. For each candidate: call backend.parse(); if it raises
   BackendUnavailable, log and continue. Any other exception is
   logged as "error" and continues too. If quality_score < min_quality,
   log "quality_low" and continue.
5. Return the first ParsedDocument that passes.
"""

from __future__ import annotations

import logging
import time

from .backends.base import BackendUnavailable, ParserBackend
from .schema import (
    BackendAttempt,
    DocFormat,
    DocProfile,
    ParsedDocument,
    ParseTrace,
)

log = logging.getLogger(__name__)


class NoBackendAvailable(RuntimeError):
    """Raised when no backend could successfully parse the document."""


class Router:
    def __init__(self, backends: list[ParserBackend]):
        # Keep only backends whose available() is True at startup.
        self._backends = [b for b in backends if b.available()]
        if not self._backends:
            raise RuntimeError("No parser backends available. At minimum PyMuPDF should be enabled.")

    # ------------------------------------------------------------------
    def build_chain(self, fmt: DocFormat, needed_tier: int) -> list[ParserBackend]:
        """Order candidates for a given document."""
        candidates = [b for b in self._backends if b.supports_format(fmt)]
        if not candidates:
            return []

        # Primary group: tier >= needed_tier, highest tier first
        primary = sorted(
            [b for b in candidates if b.tier >= needed_tier],
            key=lambda b: -b.tier,
        )
        # Fallback group: tier < needed_tier, highest tier first
        fallback = sorted(
            [b for b in candidates if b.tier < needed_tier],
            key=lambda b: -b.tier,
        )
        chain = primary + fallback

        # Ensure PyMuPDF (tier 0, pdf) is the very last safety net
        # when the doc is a PDF.
        if fmt == DocFormat.PDF:
            pymupdf_backends = [b for b in candidates if b.name == "pymupdf"]
            for b in pymupdf_backends:
                if b in chain:
                    chain.remove(b)
                chain.append(b)
        return chain

    # ------------------------------------------------------------------
    def parse(
        self,
        path: str,
        doc_id: str,
        parse_version: int,
        profile: DocProfile,
    ) -> ParsedDocument:
        chain = self.build_chain(profile.format, profile.needed_tier)
        if not chain:
            raise NoBackendAvailable(f"no backend supports format {profile.format}")

        trace = ParseTrace()
        last_error: Exception | None = None

        for backend in chain:
            t0 = time.time()
            try:
                result = backend.parse(path, doc_id, parse_version, profile)
            except BackendUnavailable as e:
                trace.record(
                    BackendAttempt(
                        backend=backend.name,
                        tier=backend.tier,
                        started_at=t0,
                        duration_ms=int((time.time() - t0) * 1000),
                        quality_score=0.0,
                        status="unavailable",
                        error_message=str(e),
                    )
                )
                last_error = e
                continue
            except Exception as e:
                log.exception("backend %s crashed on %s", backend.name, path)
                trace.record(
                    BackendAttempt(
                        backend=backend.name,
                        tier=backend.tier,
                        started_at=t0,
                        duration_ms=int((time.time() - t0) * 1000),
                        quality_score=0.0,
                        status="error",
                        error_message=str(e),
                    )
                )
                last_error = e
                continue

            quality = backend.self_check(result)
            duration_ms = int((time.time() - t0) * 1000)

            if quality >= backend.min_quality:
                trace.record(
                    BackendAttempt(
                        backend=backend.name,
                        tier=backend.tier,
                        started_at=t0,
                        duration_ms=duration_ms,
                        quality_score=quality,
                        status="ok",
                    )
                )
                result.parse_trace = trace
                return result

            trace.record(
                BackendAttempt(
                    backend=backend.name,
                    tier=backend.tier,
                    started_at=t0,
                    duration_ms=duration_ms,
                    quality_score=quality,
                    status="quality_low",
                )
            )

        raise NoBackendAvailable(f"all backends exhausted for {path}; last error: {last_error}")
