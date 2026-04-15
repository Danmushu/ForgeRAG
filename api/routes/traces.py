"""
Query trace routes for frontend visualization.

    GET  /traces                -- list recent traces
    GET  /traces/{trace_id}    -- get a single trace with full phases
    DELETE /traces/{trace_id}  -- delete a trace
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..deps import get_state
from ..state import AppState

router = APIRouter(prefix="/api/v1/traces", tags=["traces"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class TraceSummary(BaseModel):
    trace_id: str
    query: str
    timestamp: Any
    total_ms: int
    total_llm_ms: int
    total_llm_calls: int
    answer_model: str | None = None
    finish_reason: str | None = None
    citations_used: list[str]


class TraceDetail(TraceSummary):
    answer_text: str | None = None
    trace_json: dict[str, Any]
    metadata_json: dict[str, Any]


from ..schemas import PaginatedResponse


class TraceListResponse(PaginatedResponse):
    pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("", response_model=TraceListResponse)
def list_traces(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    state: AppState = Depends(get_state),
) -> TraceListResponse:
    rows = state.store.list_traces(limit=limit, offset=offset)
    items = [
        TraceSummary(
            trace_id=r["trace_id"],
            query=r["query"],
            timestamp=r["timestamp"],
            total_ms=r["total_ms"],
            total_llm_ms=r["total_llm_ms"],
            total_llm_calls=r["total_llm_calls"],
            answer_model=r.get("answer_model"),
            finish_reason=r.get("finish_reason"),
            citations_used=r.get("citations_used", []),
        )
        for r in rows
    ]
    total = state.store.count_traces()
    return TraceListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/{trace_id}", response_model=TraceDetail)
def get_trace(
    trace_id: str,
    state: AppState = Depends(get_state),
) -> TraceDetail:
    row = state.store.get_trace(trace_id)
    if not row:
        raise HTTPException(status_code=404, detail="trace not found")
    return TraceDetail(**row)


@router.delete("/{trace_id}")
def delete_trace(
    trace_id: str,
    state: AppState = Depends(get_state),
):
    row = state.store.get_trace(trace_id)
    if not row:
        raise HTTPException(status_code=404, detail="trace not found")
    state.store.delete_trace(trace_id)
    return {"deleted": trace_id}
