"""
Application state container.

Owns the long-lived collaborators (stores, pipelines, index) and
exposes them through a single attribute on the FastAPI app.

The builder path is parameterized so tests can inject fakes for
parser/tree_builder/chunker/embedder/vector_store without duplicating
wiring logic.
"""

from __future__ import annotations

import contextlib
import logging
import threading

from answering.pipeline import AnsweringPipeline
from config import AppConfig
from embedder.base import Embedder, make_embedder
from ingestion import IngestionPipeline
from ingestion.queue import IngestionQueue
from parser.blob_store import BlobStore, make_blob_store
from parser.chunker import Chunker
from parser.pipeline import ParserPipeline
from parser.tree_builder import TreeBuilder
from persistence.files import FileStore
from persistence.store import Store
from persistence.vector.base import VectorStore, make_vector_store
from retrieval.pipeline import RetrievalPipeline, build_bm25_index

log = logging.getLogger(__name__)


class AppState:
    def __init__(
        self,
        cfg: AppConfig,
        *,
        parser: ParserPipeline | None = None,
        tree_builder: TreeBuilder | None = None,
        chunker: Chunker | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        blob_store: BlobStore | None = None,
    ):
        self.cfg = cfg

        # Relational store (authoritative metadata)
        self.store = Store(cfg.persistence.relational)
        self.store.connect()
        self.store.ensure_schema()

        # Seed + apply DB settings overrides + resolve provider_id → credentials
        # This MUST happen before any component reads cfg.*
        from config.settings_manager import apply_overrides, resolve_providers, seed_defaults

        seed_defaults(cfg, self.store)
        applied = apply_overrides(cfg, self.store)
        if applied:
            log.info("applied %d DB setting overrides", applied)
        resolved = resolve_providers(cfg, self.store)
        if resolved:
            log.info("resolved %d LLM providers", resolved)

        # Blob store for figures + uploaded files
        self.blob: BlobStore = blob_store or make_blob_store(cfg.storage.to_dataclass())

        # File ingestion (files table + blob dedup)
        self.file_store = FileStore(cfg.files, self.blob, self.store)

        # Parser stack (injectable so tests can supply in-memory fakes)
        self.parser: ParserPipeline = parser or ParserPipeline.from_config(cfg)
        self.tree_builder: TreeBuilder = tree_builder or TreeBuilder(cfg.parser.tree_builder)
        self.chunker: Chunker = chunker or Chunker(cfg.parser.chunker)

        # Embedder (with optional disk cache) + vector store
        base_embedder = embedder or make_embedder(cfg.embedder)
        if embedder is None and cfg.cache.embedding_cache:
            from embedder.cached import CachedEmbedder

            self.embedder: Embedder = CachedEmbedder(
                base_embedder,
                cache_path=cfg.cache.embedding_path,
            )
        else:
            self.embedder = base_embedder
        if vector_store is not None:
            self.vector: VectorStore = vector_store
        else:
            self.vector = make_vector_store(cfg.persistence.vector, relational_store=self.store)
            self.vector.connect()
            self.vector.ensure_schema()

        # Graph store (Knowledge Graph — optional)
        self.graph_store = None
        try:
            from graph.factory import make_graph_store

            self.graph_store = make_graph_store(cfg.graph)
            log.info("graph store initialized: %s", cfg.graph.backend)

            # Wrap with entity disambiguation if enabled
            if cfg.graph.entity_disambiguation.enabled:
                try:
                    from graph.disambiguator import DisambiguatingGraphStore, EntityDisambiguator

                    disambiguator = EntityDisambiguator(
                        embedder=self.embedder,
                        threshold=cfg.graph.entity_disambiguation.similarity_threshold,
                        candidate_top_k=cfg.graph.entity_disambiguation.candidate_top_k,
                    )
                    existing = self.graph_store.get_all_entities()
                    disambiguator.load_existing(existing)
                    self.graph_store = DisambiguatingGraphStore(self.graph_store, disambiguator)
                    log.info(
                        "entity disambiguation enabled (threshold=%.2f, cached=%d)",
                        cfg.graph.entity_disambiguation.similarity_threshold,
                        len(existing),
                    )
                except Exception as e:
                    log.warning("entity disambiguation init failed: %s", e)
        except Exception as e:
            log.warning("graph store not available: %s", e, exc_info=True)

        # Ingestion orchestrator
        self.ingestion = IngestionPipeline(
            file_store=self.file_store,
            parser=self.parser,
            tree_builder=self.tree_builder,
            chunker=self.chunker,
            relational_store=self.store,
            vector_store=self.vector,
            embedder=self.embedder,
            graph_store=self.graph_store,
            kg_extraction_cfg=cfg.retrieval.kg_extraction,
        )

        # Background ingestion queue
        self.ingest_queue = IngestionQueue(
            self.ingestion,
            max_workers=cfg.parser.ingest_max_workers,
            on_complete=self._on_ingest_complete,
        )
        self.ingest_queue.start()

        # Re-queue documents that were stuck mid-ingestion when the
        # process last exited (crash, restart, worker recycled by uvicorn).
        self.ingest_queue.recover_stuck(self.store)

        # BM25 index is lazy -- built on first /ask
        self._init_lock = threading.RLock()
        self._bm25 = None
        self._retrieval: RetrievalPipeline | None = None
        self._answering: AnsweringPipeline | None = None

    # ------------------------------------------------------------------
    def _ensure_retrieval(self) -> RetrievalPipeline:
        if self._retrieval is not None:
            return self._retrieval
        with self._init_lock:
            if self._retrieval is not None:
                return self._retrieval
            cache_path = self.cfg.cache.bm25_path if self.cfg.cache.bm25_persistence else ""
            self._bm25 = build_bm25_index(
                self.store,
                self.cfg.retrieval.bm25,
                cache_path=cache_path,
            )

            # Build LLM tree navigator if configured
            tree_nav = None
            tp = self.cfg.retrieval.tree_path
            if tp.llm_nav_enabled:
                from retrieval.tree_navigator import LLMTreeNavigator

                tree_nav = LLMTreeNavigator(
                    model=tp.nav.model,
                    api_key=tp.nav.api_key,
                    api_key_env=tp.nav.api_key_env,
                    api_base=tp.nav.api_base,
                    temperature=tp.nav.temperature,
                    max_tokens=tp.nav.max_tokens,
                    timeout=tp.nav.timeout,
                    max_nodes=tp.nav.max_nodes,
                    system_prompt=tp.nav.system_prompt,
                )

            self._retrieval = RetrievalPipeline(
                self.cfg.retrieval,
                embedder=self.embedder,
                vector_store=self.vector,
                relational_store=self.store,
                bm25_index=self._bm25,
                tree_navigator=tree_nav,
                graph_store=self.graph_store,
            )
            return self._retrieval

    def _ensure_answering(self) -> AnsweringPipeline:
        if self._answering is not None:
            return self._answering
        with self._init_lock:
            if self._answering is not None:
                return self._answering
            retrieval = self._ensure_retrieval()
            self._answering = AnsweringPipeline(
                self.cfg.answering,
                retrieval=retrieval,
                store=self.store,
            )
            return self._answering

    # ------------------------------------------------------------------
    @property
    def retrieval(self) -> RetrievalPipeline:
        return self._ensure_retrieval()

    @property
    def answering(self) -> AnsweringPipeline:
        return self._ensure_answering()

    # ------------------------------------------------------------------
    def refresh_bm25(self, *, force_rebuild: bool = True) -> None:
        """Rebuild BM25 and optionally persist to disk cache."""
        cache_path = self.cfg.cache.bm25_path if self.cfg.cache.bm25_persistence else None
        new_bm25 = build_bm25_index(
            self.store,
            self.cfg.retrieval.bm25,
            force_rebuild=force_rebuild,
            cache_path=cache_path or "",
        )
        with self._init_lock:
            self._bm25 = new_bm25
            if self._retrieval is not None:
                self._retrieval.bm25 = new_bm25

    # ------------------------------------------------------------------
    def _on_ingest_complete(self, doc_id: str, error: Exception | None) -> None:
        """Called from worker thread after each ingestion job finishes."""
        if error is None:
            try:
                self.refresh_bm25()
            except Exception:
                log.exception("post-ingest bm25 refresh failed")

            # Auto-rebuild KG communities when the queue drains.
            # Leiden runs on the full graph, so we only trigger it once
            # after the last document finishes — not after every single one.
            if self.ingest_queue.pending_count == 0:
                self._maybe_rebuild_communities()

    # ------------------------------------------------------------------
    def _maybe_rebuild_communities(self) -> None:
        """Rebuild KG communities after ingestion queue drains.

        Conditions (all must be true):
            1. graph_store is available
            2. KG extraction is enabled
            3. community_detection is enabled
            4. Graph has entities (non-empty)

        Runs Leiden clustering → LLM summarization → embedding → store.
        Skips silently if any precondition is not met.
        """
        gs = self.graph_store
        if gs is None:
            return
        kg_cfg = self.cfg.retrieval.kg_extraction
        if not (kg_cfg and kg_cfg.enabled):
            return
        comm_cfg = self.cfg.graph.community_detection
        if not (comm_cfg and comm_cfg.enabled):
            return

        try:
            stats = gs.stats()
            if stats.get("entities", 0) < comm_cfg.min_community_size:
                return

            log.info("auto-detecting KG communities (queue drained) ...")
            communities = gs.detect_communities(resolution=comm_cfg.resolution)
            communities = [c for c in communities if len(c.entity_ids) >= comm_cfg.min_community_size]

            if not communities:
                log.info("community detection: 0 communities found")
                return

            # LLM summarization + embedding.
            # Fall back to KG extraction's LLM config if community_detection
            # doesn't have its own — so users only need to configure one provider.
            from config.auth import resolve_api_key
            from graph.community_summarizer import CommunitySummarizer

            model = comm_cfg.model
            c_api_key = comm_cfg.api_key
            c_api_key_env = comm_cfg.api_key_env
            c_api_base = comm_cfg.api_base
            if not c_api_key and not c_api_key_env and not c_api_base:
                # Inherit from KG extraction config
                c_api_key = kg_cfg.api_key
                c_api_key_env = kg_cfg.api_key_env
                c_api_base = kg_cfg.api_base
                model = kg_cfg.model or model

            api_key = resolve_api_key(
                api_key=c_api_key,
                api_key_env=c_api_key_env,
                required=False,
                context="community_auto_summarizer",
            )
            summarizer = CommunitySummarizer(
                model=model,
                api_key=api_key,
                api_base=c_api_base,
                embedder=self.embedder,
                timeout=comm_cfg.timeout,
                max_workers=comm_cfg.max_workers,
            )
            communities = summarizer.summarize_communities(communities, gs)

            for c in communities:
                gs.upsert_community(c)

            log.info(
                "auto-community detection done: %d communities, %d with summaries",
                len(communities),
                sum(1 for c in communities if c.summary and c.summary != c.title),
            )
        except Exception:
            log.exception("auto-community detection failed")

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        # Stop ingestion queue
        try:
            self.ingest_queue.shutdown()
        except Exception:
            log.exception("ingestion queue shutdown failed")
        # Save embedding cache
        if hasattr(self.embedder, "save"):
            with contextlib.suppress(Exception):
                self.embedder.save()
        # Save BM25 cache
        if self._bm25 is not None:
            try:
                cache_path = self.cfg.cache.bm25_path if self.cfg.cache.bm25_persistence else None
                if cache_path:
                    self._bm25.save(cache_path)
            except Exception:
                pass
        # Close graph store
        if self.graph_store is not None:
            try:
                self.graph_store.close()
            except Exception:
                log.exception("graph store close failed")
        try:
            self.store.close()
        except Exception:
            log.exception("store close failed")
        try:
            self.vector.close()
        except Exception:
            log.exception("vector close failed")
