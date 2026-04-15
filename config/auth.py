"""
API key resolution helper.

Used by every litellm-backed component (embedder, generator, reranker).
Single source of truth for the precedence:

    1. cfg.api_key         (plaintext in yaml — simple, dev-friendly)
    2. cfg.api_key_env     (name of env var — production-friendly)
    3. None                (some backends like Ollama don't need a key)
"""

from __future__ import annotations

import os


def resolve_api_key(
    *,
    api_key: str | None = None,
    api_key_env: str | None = None,
    required: bool = False,
    context: str = "",
) -> str | None:
    """
    Return the resolved API key string, or None if not configured.
    Raises RuntimeError if `required=True` and no key is found.
    """
    # 1. Direct value
    if api_key:
        return api_key

    # 2. Env var
    if api_key_env:
        val = os.environ.get(api_key_env)
        if val:
            return val
        if required:
            raise RuntimeError(
                f"{context}: api_key_env={api_key_env!r} is set in config "
                f"but the environment variable is not defined. "
                f"Either set the env var or use api_key directly in the yaml."
            )
        return None

    if required:
        raise RuntimeError(f"{context}: no api_key or api_key_env configured")
    return None
