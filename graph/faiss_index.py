"""
Lightweight FAISS-backed vector index with automatic fallback.

Used internally by NetworkXGraphStore and EntityDisambiguator for
fast approximate-nearest-neighbour cosine similarity search.

When ``faiss`` (``faiss-cpu``) is installed the index uses
``faiss.IndexFlatIP`` on L2-normalised vectors (= exact cosine
similarity).  If FAISS is not available it falls back to a pure-Python
brute-force implementation — correct but O(n) per query.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

try:
    import faiss
    import numpy as np

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class VectorIndex:
    """Add / remove / search vectors by string key.

    Parameters
    ----------
    dim : int or None
        Embedding dimension.  Can be ``None`` at construction time; it will
        be inferred from the first ``add`` call.
    """

    def __init__(self, dim: int | None = None) -> None:
        self._lock = threading.RLock()
        self._dim = dim
        # ordered key list — position matches FAISS internal id
        self._keys: list[str] = []
        # key → position (for fast remove / dedup)
        self._key2pos: dict[str, int] = {}
        # FAISS index (created lazily on first add)
        self._index: faiss.IndexFlatIP | None = None  # type: ignore[name-defined]
        # fallback raw vectors when faiss unavailable
        self._raw: list[list[float]] = []

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._keys)

    def add(self, key: str, vector: list[float]) -> None:
        """Insert or replace *key* with *vector*."""
        with self._lock:
            if not vector:
                return
            if self._dim is None:
                self._dim = len(vector)
            if key in self._key2pos:
                # remove old entry first, then re-add
                self._remove_unlocked(key)
            self._keys.append(key)
            self._key2pos[key] = len(self._keys) - 1
            if _HAS_FAISS:
                self._ensure_index()
                vec_np = self._normalise(vector)
                self._index.add(vec_np)  # type: ignore[union-attr]
            else:
                self._raw.append(vector)

    def add_batch(self, keys: list[str], vectors: list[list[float]]) -> None:
        """Bulk insert — significantly faster than repeated single adds."""
        with self._lock:
            if not keys:
                return
            if self._dim is None and vectors:
                self._dim = len(vectors[0])

            new_keys: list[str] = []
            new_vecs: list[list[float]] = []
            for k, v in zip(keys, vectors, strict=False):
                if not v:
                    continue
                if k in self._key2pos:
                    self._remove_unlocked(k)
                new_keys.append(k)
                new_vecs.append(v)

            if not new_keys:
                return

            base = len(self._keys)
            self._keys.extend(new_keys)
            for i, k in enumerate(new_keys):
                self._key2pos[k] = base + i

            if _HAS_FAISS:
                self._ensure_index()
                mat = self._normalise_batch(new_vecs)
                self._index.add(mat)  # type: ignore[union-attr]
            else:
                self._raw.extend(new_vecs)

    def remove(self, key: str) -> None:
        """Remove *key* from the index.  Triggers a full rebuild."""
        with self._lock:
            self._remove_unlocked(key)

    def search(self, query: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """Return up to *top_k* ``(key, cosine_similarity)`` pairs, descending."""
        with self._lock:
            if not self._keys or not query:
                return []
            if _HAS_FAISS:
                return self._search_faiss(query, top_k)
            else:
                return self._search_brute(query, top_k)

    def clear(self) -> None:
        with self._lock:
            self._keys.clear()
            self._key2pos.clear()
            self._raw.clear()
            self._index = None

    # ------------------------------------------------------------------
    # FAISS internals
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        """Create the FAISS index if it doesn't exist yet."""
        if self._index is None and self._dim is not None:
            self._index = faiss.IndexFlatIP(self._dim)

    def _normalise(self, vec: list[float]) -> np.ndarray:
        """L2-normalise a single vector → contiguous (1, dim) float32 array."""
        a = np.ascontiguousarray(np.array(vec, dtype=np.float32).reshape(1, -1))
        faiss.normalize_L2(a)
        return a

    def _normalise_batch(self, vecs: list[list[float]]) -> np.ndarray:
        a = np.ascontiguousarray(np.array(vecs, dtype=np.float32))
        faiss.normalize_L2(a)
        return a

    def _search_faiss(
        self,
        query: list[float],
        top_k: int,
    ) -> list[tuple[str, float]]:
        q = self._normalise(query)
        k = min(top_k, self._index.ntotal)  # type: ignore[union-attr]
        if k == 0:
            return []
        D, I = self._index.search(q, k)  # type: ignore[union-attr]
        results: list[tuple[str, float]] = []
        for score, idx in zip(D[0], I[0], strict=False):
            if idx < 0 or idx >= len(self._keys):
                continue
            results.append((self._keys[idx], float(score)))
        return results

    # ------------------------------------------------------------------
    # Brute-force fallback (no numpy / no faiss)
    # ------------------------------------------------------------------

    def _search_brute(
        self,
        query: list[float],
        top_k: int,
    ) -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        for i, vec in enumerate(self._raw):
            sim = _cosine(query, vec)
            scored.append((self._keys[i], sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Remove helper — must hold _lock
    # ------------------------------------------------------------------

    def _remove_unlocked(self, key: str) -> None:
        if key not in self._key2pos:
            return
        pos = self._key2pos.pop(key)
        self._keys.pop(pos)
        # Re-index positions
        self._key2pos = {k: i for i, k in enumerate(self._keys)}

        if _HAS_FAISS:
            # Rebuild FAISS index from remaining vectors.
            # Use reconstruct() (public API) instead of rev_swig_ptr.
            if self._index is not None and self._index.ntotal > 0:
                n = self._index.ntotal
                all_vecs = np.empty((n, self._dim), dtype=np.float32)
                for i in range(n):
                    all_vecs[i] = self._index.reconstruct(i)
                remaining = np.delete(all_vecs, pos, axis=0)
                self._index = faiss.IndexFlatIP(self._dim)
                if len(remaining) > 0:
                    # Vectors are already L2-normalised in the index
                    self._index.add(np.ascontiguousarray(remaining))
            else:
                self._index = faiss.IndexFlatIP(self._dim) if self._dim else None
        else:
            self._raw.pop(pos)


def _cosine(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity fallback."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)
