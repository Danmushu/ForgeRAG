"""
Benchmark API endpoints.

    POST   /api/v1/benchmark/start    — start a benchmark run
    POST   /api/v1/benchmark/cancel   — cancel a running benchmark
    GET    /api/v1/benchmark/status   — poll current status / results
    GET    /api/v1/benchmark/report   — download full report as JSON
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from benchmark.report import build_report
from benchmark.runner import BenchmarkRunner

from ..deps import get_state
from ..state import AppState

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/benchmark", tags=["benchmark"])

_runner = BenchmarkRunner()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class BenchmarkStartRequest(BaseModel):
    num_questions: int = Field(default=30, ge=5, le=200)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/start")
def start_benchmark(req: BenchmarkStartRequest, state: AppState = Depends(get_state)):
    if _runner.running:
        raise HTTPException(409, "A benchmark is already running")
    try:
        run_id = _runner.start(
            cfg=state.cfg,
            store=state.store,
            answering=state.answering,
            num_questions=req.num_questions,
        )
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"run_id": run_id}


@router.post("/cancel")
def cancel_benchmark():
    _runner.cancel()
    return {"ok": True}


@router.get("/status")
def benchmark_status():
    return _runner.get_status()


@router.get("/report")
def download_report():
    status = _runner.get_status()
    if status["status"] != "done":
        raise HTTPException(400, "Benchmark not complete yet")
    report = build_report(status)
    content = json.dumps(report, ensure_ascii=False, indent=2)
    return Response(
        content=content,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="forgerag_benchmark_{status["run_id"]}.json"',
        },
    )
