"""
LLM Providers — pluggable model registry.

CRUD for managing chat / embedding / reranker endpoints.
The API never exposes raw API keys in GET responses (only a boolean flag).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_state
from ..schemas import LLMProviderCreate, LLMProviderOut, LLMProviderUpdate
from ..state import AppState

router = APIRouter(prefix="/api/v1/llm-providers", tags=["llm-providers"])


def _to_out(row: dict) -> LLMProviderOut:
    return LLMProviderOut(
        id=row["id"],
        name=row["name"],
        provider_type=row["provider_type"],
        api_base=row["api_base"],
        model_name=row["model_name"],
        api_key_set=bool(row.get("api_key")),
        is_default=row.get("is_default", False),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


@router.get("", response_model=list[LLMProviderOut])
def list_providers(
    provider_type: str | None = None,
    state: AppState = Depends(get_state),
):
    """List all registered LLM providers, optionally filtered by type."""
    rows = state.store.list_llm_providers(provider_type=provider_type)
    return [_to_out(r) for r in rows]


@router.get("/{provider_id}", response_model=LLMProviderOut)
def get_provider(provider_id: str, state: AppState = Depends(get_state)):
    row = state.store.get_llm_provider(provider_id)
    if not row:
        raise HTTPException(404, f"provider {provider_id!r} not found")
    return _to_out(row)


@router.post("", response_model=LLMProviderOut, status_code=201)
def create_provider(body: LLMProviderCreate, state: AppState = Depends(get_state)):
    if body.provider_type not in ("chat", "embedding", "reranker", "vlm"):
        raise HTTPException(422, "provider_type must be chat, embedding, reranker, or vlm")
    existing = state.store.get_llm_provider_by_name(body.name)
    if existing:
        raise HTTPException(409, f"provider name {body.name!r} already exists")
    record = {
        "id": uuid.uuid4().hex[:16],
        "name": body.name,
        "provider_type": body.provider_type,
        "api_base": body.api_base,
        "model_name": body.model_name,
        "api_key": body.api_key,
        "is_default": body.is_default,
    }
    state.store.upsert_llm_provider(record)
    row = state.store.get_llm_provider(record["id"])
    return _to_out(row)


@router.put("/{provider_id}", response_model=LLMProviderOut)
def update_provider(
    provider_id: str,
    body: LLMProviderUpdate,
    state: AppState = Depends(get_state),
):
    existing = state.store.get_llm_provider(provider_id)
    if not existing:
        raise HTTPException(404, f"provider {provider_id!r} not found")
    if body.provider_type and body.provider_type not in ("chat", "embedding", "reranker", "vlm"):
        raise HTTPException(422, "provider_type must be chat, embedding, reranker, or vlm")
    updates = body.model_dump(exclude_none=True)
    if updates:
        updates["id"] = provider_id
        state.store.upsert_llm_provider(updates)

    # Re-resolve all provider_id references so live config picks up changes
    from config.settings_manager import resolve_providers

    resolve_providers(state.cfg, state.store)
    state._retrieval = None
    state._answering = None

    row = state.store.get_llm_provider(provider_id)
    return _to_out(row)


@router.delete("/{provider_id}")
def delete_provider(provider_id: str, state: AppState = Depends(get_state)):
    existing = state.store.get_llm_provider(provider_id)
    if not existing:
        raise HTTPException(404, f"provider {provider_id!r} not found")
    state.store.delete_llm_provider(provider_id)

    # Re-resolve (the deleted provider's fields will no longer match)
    from config.settings_manager import resolve_providers

    resolve_providers(state.cfg, state.store)
    state._retrieval = None
    state._answering = None

    return {"deleted": provider_id}
