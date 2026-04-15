"""
Local sentence-transformers embedder.

Loads a HuggingFace sentence-transformers model into memory (CPU
or CUDA) and runs encode() in batches. The model is loaded lazily
so importing this module does not drag torch into environments
that never use it.
"""

from __future__ import annotations

import logging
from typing import Any

from config import EmbedderConfig
from parser.schema import Chunk

from .base import chunk_to_embedding_text

log = logging.getLogger(__name__)


class SentenceTransformersEmbedder:
    backend = "sentence_transformers"

    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        assert cfg.sentence_transformers is not None
        self.inner = cfg.sentence_transformers
        self._model: Any = None

    @property
    def dimension(self):
        return self.cfg.dimension

    @property
    def batch_size(self):
        return self.cfg.batch_size

    # ------------------------------------------------------------------
    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "SentenceTransformersEmbedder requires sentence-transformers: pip install sentence-transformers"
            ) from e

        kwargs: dict[str, Any] = {
            "device": self.inner.device,
            "trust_remote_code": self.inner.trust_remote_code,
        }
        if self.inner.cache_folder:
            kwargs["cache_folder"] = self.inner.cache_folder
        log.info("loading sentence-transformers model: %s", self.inner.model_name)
        self._model = SentenceTransformer(self.inner.model_name, **kwargs)

        # Sanity check declared vs actual dimension
        actual = self._model.get_sentence_embedding_dimension()
        if actual != self.dimension:
            raise RuntimeError(
                f"model {self.inner.model_name} produces dim={actual}, config.dimension={self.dimension}"
            )
        return self._model

    # ------------------------------------------------------------------
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.inner.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # numpy -> plain lists so we're detached from torch/numpy downstream
        return [v.tolist() for v in vectors]

    # ------------------------------------------------------------------
    def embed_chunks(self, chunks: list[Chunk]) -> dict[str, list[float]]:
        items: list[tuple[str, str]] = []
        for c in chunks:
            text = chunk_to_embedding_text(c)
            if text:
                items.append((c.chunk_id, text))
        if not items:
            return {}
        ids = [cid for cid, _ in items]
        texts = [t for _, t in items]
        vectors = self.embed_texts(texts)
        return dict(zip(ids, vectors, strict=False))
