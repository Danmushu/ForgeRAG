"""
Parser pipeline -- the single public entry point.

Wires together: probe -> router -> normalizer.

Typical usage:

    from config import load_config
    from parser.pipeline import ParserPipeline

    cfg = load_config("forgerag.yaml")
    pipeline = ParserPipeline.from_config(cfg)
    doc = pipeline.parse("paper.pdf", doc_id="doc_abc", parse_version=1)

The pipeline owns the backend instances and the BlobStore; it is
cheap to construct but should be reused across documents so that
boto3/oss2 clients and model handles are shared.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import AppConfig

from .backends.base import ParserBackend
from .backends.pymupdf import PyMuPDFBackend
from .blob_store import BlobStore, make_blob_store
from .normalizer import normalize
from .probe import probe
from .router import Router
from .schema import ParsedDocument

log = logging.getLogger(__name__)


class ParserPipeline:
    def __init__(
        self,
        cfg: AppConfig,
        blob_store: BlobStore,
        backends: list[ParserBackend],
    ):
        self.cfg = cfg
        self.blob_store = blob_store
        self.router = Router(backends)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: AppConfig) -> ParserPipeline:
        """Build a pipeline from a validated AppConfig."""
        blob_store = make_blob_store(cfg.storage.to_dataclass())
        backends = _build_backends(cfg, blob_store)
        return cls(cfg=cfg, blob_store=blob_store, backends=backends)

    # ------------------------------------------------------------------
    def parse(
        self,
        path: str | Path,
        *,
        doc_id: str,
        parse_version: int = 1,
    ) -> ParsedDocument:
        """
        Parse a single document end-to-end.

        Raises:
            ValueError       -- unsupported format
            NoBackendAvailable -- no backend could handle the file
        """
        path_str = str(path)
        log.info("parse start doc_id=%s path=%s", doc_id, path_str)

        # 1. Layer-0 probe
        profile = probe(path_str, self.cfg.parser.probe)
        log.info(
            "probe doc_id=%s format=%s pages=%d needed_tier=%d complexity=%s",
            doc_id,
            profile.format.value,
            profile.page_count,
            profile.needed_tier,
            profile.complexity.value,
        )

        # 2. Router -> backend chain -> parse
        result = self.router.parse(
            path=path_str,
            doc_id=doc_id,
            parse_version=parse_version,
            profile=profile,
        )
        log.info(
            "parse done doc_id=%s backend=%s quality=%.3f blocks=%d",
            doc_id,
            result.parse_trace.final_backend,
            result.parse_trace.final_quality or 0.0,
            len(result.blocks),
        )

        # 3. Normalizer (always runs; controlled by config switches)
        result = normalize(result, self.cfg.parser.normalize)
        excluded = sum(1 for b in result.blocks if b.excluded)
        log.info(
            "normalize done doc_id=%s excluded_blocks=%d reading_blocks=%d",
            doc_id,
            excluded,
            len(result.blocks) - excluded,
        )
        return result


# ---------------------------------------------------------------------------
# Backend wiring
# ---------------------------------------------------------------------------


def _build_backends(cfg: AppConfig, blob_store: BlobStore) -> list[ParserBackend]:
    """
    Instantiate every backend whose config section has enabled=True.
    Layer 1/2/Docling backends are imported lazily so missing optional
    dependencies don't crash the pipeline at import time.
    """
    backends: list[ParserBackend] = []
    pcfg = cfg.parser.backends

    # Layer 0 -- PyMuPDF (always the final fallback)
    if pcfg.pymupdf.enabled:
        backends.append(PyMuPDFBackend(pcfg.pymupdf, blob_store))

    # Layer 1 -- MinerU (lazy import)
    if pcfg.mineru.enabled:
        try:
            from .backends.mineru import MinerUBackend  # type: ignore

            backends.append(MinerUBackend(pcfg.mineru, blob_store))
        except ImportError as e:
            log.warning("MinerU enabled in config but import failed: %s", e)

    # Layer 2 -- VLM (lazy import)
    if pcfg.vlm.enabled:
        try:
            from .backends.vlm import VLMBackend  # type: ignore

            backends.append(VLMBackend(pcfg.vlm, blob_store))
        except ImportError as e:
            log.warning("VLM enabled in config but import failed: %s", e)

    # Office / HTML -- Docling (lazy import)
    if pcfg.docling.enabled:
        try:
            from .backends.docling import DoclingBackend  # type: ignore

            backends.append(DoclingBackend(pcfg.docling, blob_store))
        except ImportError as e:
            log.warning("Docling enabled in config but import failed: %s", e)

    return backends


# ---------------------------------------------------------------------------
# Convenience functional API
# ---------------------------------------------------------------------------


_default_pipeline: ParserPipeline | None = None


def parse(
    path: str | Path,
    *,
    doc_id: str,
    parse_version: int = 1,
    cfg: AppConfig | None = None,
) -> ParsedDocument:
    """
    One-shot parse using a cached default pipeline. For production
    use prefer constructing ParserPipeline explicitly and reusing it.
    """
    global _default_pipeline
    if _default_pipeline is None or cfg is not None:
        from config import load_config

        _default_pipeline = ParserPipeline.from_config(cfg or load_config())
    return _default_pipeline.parse(path, doc_id=doc_id, parse_version=parse_version)
