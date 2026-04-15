"""
FastAPI dependency providers.

Routes receive the AppState through these so the container lives
as a request-scoped dependency and tests can override it via
`app.dependency_overrides`.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from .state import AppState


def get_state(request: Request) -> AppState:
    state = getattr(request.app.state, "app", None)
    if state is None:
        raise HTTPException(status_code=503, detail="app state not initialized")
    return state


# Type alias-ish: routes use `state: AppState = Depends(get_state)`
StateDep = Depends(get_state)
