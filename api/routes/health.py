"""GET /api/v1/health"""

from fastapi import APIRouter, Depends

from ..deps import get_state
from ..schemas import HealthResponse
from ..state import AppState

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(state: AppState = Depends(get_state)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        components={
            "relational": state.store.backend,
            "vector": state.vector.backend,
            "blob": state.blob.mode,
            "embedder": state.embedder.backend,
        },
        counts={
            "documents": state.store.count_documents(),
            "files": state.store.count_files(),
        },
    )
