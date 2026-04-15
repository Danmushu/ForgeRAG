"""
/api/v1/files — file upload and management
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse, Response

from ..deps import get_state
from ..schemas import FileOut, PaginatedResponse, UploadUrlRequest
from ..state import AppState

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/files", tags=["files"])


def _validate_url(url: str):
    """Reject non-HTTP schemes and URLs that resolve to private/reserved IPs (SSRF protection)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(400, "Only http/https URLs are allowed")
    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(400, "Invalid URL")
    try:
        for info in socket.getaddrinfo(hostname, None):
            addr = ipaddress.ip_address(info[4][0])
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                raise HTTPException(400, "URL resolves to private/reserved IP")
    except (socket.gaierror, ValueError):
        pass  # DNS resolution failed, let the actual fetch handle it


def _to_out(row: dict) -> FileOut:
    return FileOut(**{k: row[k] for k in FileOut.model_fields if k in row})


@router.post("", response_model=FileOut, status_code=201)
async def upload_file(
    file: UploadFile = File(...),
    original_name: str | None = Form(None),
    mime_type: str | None = Form(None),
    state: AppState = Depends(get_state),
) -> FileOut:
    name = original_name or file.filename or "upload.bin"
    mime = mime_type or file.content_type
    data = await file.read()
    files_cfg = getattr(state.cfg, "files", None)
    max_bytes = files_cfg.max_bytes if files_cfg and files_cfg.max_bytes else 524288000
    if len(data) > max_bytes:
        raise HTTPException(413, f"File too large: {len(data)} bytes exceeds limit of {max_bytes}")
    try:
        record = state.file_store.store(data, original_name=name, mime_type=mime)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _to_out(record)


@router.post("/from-url", response_model=FileOut, status_code=201)
def upload_from_url(
    req: UploadUrlRequest,
    state: AppState = Depends(get_state),
) -> FileOut:
    _validate_url(req.url)
    try:
        record = state.file_store.store_from_url(
            req.url,
            original_name=req.original_name,
            mime_type=req.mime_type,
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    return _to_out(record)


@router.get("", response_model=PaginatedResponse)
def list_files(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    state: AppState = Depends(get_state),
):
    rows = state.store.list_files(limit=limit, offset=offset)
    total = state.store.count_files()
    return PaginatedResponse(
        items=[_to_out(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{file_id}", response_model=FileOut)
def get_file(file_id: str, state: AppState = Depends(get_state)):
    row = state.store.get_file(file_id)
    if not row:
        raise HTTPException(404, "file not found")
    return _to_out(row)


@router.get("/{file_id}/download")
def download_file(file_id: str, state: AppState = Depends(get_state)):
    row = state.store.get_file(file_id)
    if not row:
        raise HTTPException(404, "file not found")
    url = state.blob.url_for(row["storage_key"])
    if url and url.startswith(("http://", "https://")):
        return RedirectResponse(url, status_code=302)
    try:
        data = state.blob.get(row["storage_key"])
    except FileNotFoundError:
        raise HTTPException(404, "blob missing")
    safe_name = row["display_name"].replace('"', "_").replace("\n", "_").replace("\r", "_")
    return Response(
        content=data,
        media_type=row["mime_type"],
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


@router.get("/{file_id}/preview")
def preview_file(file_id: str, state: AppState = Depends(get_state)):
    """Serve file inline for iframe/embed — never redirect, always stream."""
    row = state.store.get_file(file_id)
    if not row:
        raise HTTPException(404, "file not found")
    try:
        data = state.blob.get(row["storage_key"])
    except FileNotFoundError:
        raise HTTPException(404, "blob missing")
    # HTTP headers only support latin-1.  For non-ASCII filenames
    # (Chinese, Japanese, etc.) use RFC 5987 UTF-8 encoding.
    from urllib.parse import quote

    raw_name = row["display_name"].replace('"', "_").replace("\n", "_").replace("\r", "_")
    try:
        raw_name.encode("latin-1")
        disposition = f'inline; filename="{raw_name}"'
    except UnicodeEncodeError:
        encoded = quote(raw_name, safe="")
        disposition = f"inline; filename*=UTF-8''{encoded}"
    return Response(
        content=data,
        media_type=row["mime_type"],
        headers={
            "Content-Disposition": disposition,
            "Cache-Control": "private, max-age=3600",
        },
    )


@router.delete("/{file_id}", status_code=204)
def delete_file(file_id: str, state: AppState = Depends(get_state)):
    row = state.store.get_file(file_id)
    if not row:
        raise HTTPException(404, "file not found")
    state.store.delete_file(file_id)
