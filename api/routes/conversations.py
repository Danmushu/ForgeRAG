"""
/api/v1/conversations — multi-turn chat management

    GET    /api/v1/conversations                       list all
    POST   /api/v1/conversations                       create empty
    GET    /api/v1/conversations/{id}                   detail + message count
    DELETE /api/v1/conversations/{id}                   delete (cascade messages)
    PATCH  /api/v1/conversations/{id}                   update title
    GET    /api/v1/conversations/{id}/messages          message history
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..deps import get_state
from ..schemas import ConversationOut, MessageOut, PaginatedResponse
from ..state import AppState

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


class CreateConversationRequest(BaseModel):
    title: str | None = None


class UpdateConversationRequest(BaseModel):
    title: str


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


@router.get("", response_model=PaginatedResponse)
def list_conversations(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    state: AppState = Depends(get_state),
):
    rows = state.store.list_conversations(limit=limit, offset=offset)
    total = state.store.count_conversations()
    items = []
    for r in rows:
        r["message_count"] = state.store.count_messages(r["conversation_id"])
        items.append(ConversationOut(**{k: r[k] for k in ConversationOut.model_fields if k in r}))
    return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)


@router.post("", response_model=ConversationOut, status_code=201)
def create_conversation(
    req: CreateConversationRequest = CreateConversationRequest(),
    state: AppState = Depends(get_state),
):
    cid = uuid4().hex
    state.store.create_conversation(
        {
            "conversation_id": cid,
            "title": req.title,
        }
    )
    row = state.store.get_conversation(cid)
    return ConversationOut(**row, message_count=0)


@router.get("/{conversation_id}", response_model=ConversationOut)
def get_conversation(
    conversation_id: str,
    state: AppState = Depends(get_state),
):
    row = state.store.get_conversation(conversation_id)
    if not row:
        raise HTTPException(404, "conversation not found")
    row["message_count"] = state.store.count_messages(conversation_id)
    return ConversationOut(**{k: row[k] for k in ConversationOut.model_fields if k in row})


@router.patch("/{conversation_id}", response_model=ConversationOut)
def update_conversation(
    conversation_id: str,
    req: UpdateConversationRequest,
    state: AppState = Depends(get_state),
):
    row = state.store.get_conversation(conversation_id)
    if not row:
        raise HTTPException(404, "conversation not found")
    state.store.update_conversation(conversation_id, title=req.title)
    return get_conversation(conversation_id, state)


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: str,
    state: AppState = Depends(get_state),
):
    row = state.store.get_conversation(conversation_id)
    if not row:
        raise HTTPException(404, "conversation not found")
    state.store.delete_conversation(conversation_id)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
def list_messages(
    conversation_id: str,
    limit: int = Query(100, ge=1, le=500),
    state: AppState = Depends(get_state),
):
    row = state.store.get_conversation(conversation_id)
    if not row:
        raise HTTPException(404, "conversation not found")
    msgs = state.store.get_messages(conversation_id, limit=limit)
    return [MessageOut(**{k: m[k] for k in MessageOut.model_fields if k in m}) for m in msgs]


class AddMessageRequest(BaseModel):
    role: str = Field(pattern=r"^(user|assistant)$")
    content: str


@router.post("/{conversation_id}/messages", response_model=MessageOut, status_code=201)
def add_message(
    conversation_id: str,
    req: AddMessageRequest,
    state: AppState = Depends(get_state),
):
    """Manually add a message to a conversation (used for preset Q&A)."""
    row = state.store.get_conversation(conversation_id)
    if not row:
        raise HTTPException(404, "conversation not found")
    mid = uuid4().hex
    state.store.add_message(
        {
            "message_id": mid,
            "conversation_id": conversation_id,
            "role": req.role,
            "content": req.content,
        }
    )
    return MessageOut(
        message_id=mid,
        conversation_id=conversation_id,
        role=req.role,
        content=req.content,
    )
