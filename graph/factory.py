"""
Factory for creating a :class:`GraphStore` from configuration.

Usage::

    from graph import make_graph_store
    store = make_graph_store(cfg.graph)
"""

from __future__ import annotations

import os
from typing import Any

from .base import GraphStore


def _resolve_env(password: str, password_env: str | None) -> str:
    """Return *password* if non-empty, otherwise read from *password_env*."""
    if password:
        return password
    if password_env:
        val = os.environ.get(password_env, "")
        if not val:
            raise ValueError(f"Graph store password is empty and env var {password_env!r} is not set")
        return val
    return ""


def make_graph_store(cfg: Any) -> GraphStore:
    """Instantiate the appropriate :class:`GraphStore` backend.

    Parameters
    ----------
    cfg:
        An object (typically a Pydantic model) with at least a ``backend``
        attribute (``"networkx"`` or ``"neo4j"``), and optional ``networkx``
        / ``neo4j`` sub-config objects.
    """
    if cfg.backend == "neo4j":
        from .neo4j_store import Neo4jGraphStore

        neo = cfg.neo4j
        return Neo4jGraphStore(
            uri=neo.uri,
            user=neo.user,
            password=_resolve_env(neo.password, getattr(neo, "password_env", None)),
            database=neo.database,
        )
    else:
        from .networkx_store import NetworkXGraphStore

        path = cfg.networkx.path if getattr(cfg, "networkx", None) is not None else "./storage/kg.json"
        return NetworkXGraphStore(path=path)
