"""
File ingestion configuration.

The actual storage backend is controlled by `storage:` (Local/S3/OSS).
This section controls how files are hashed, sharded, and validated
before being handed to the blob store.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FilesConfig(BaseModel):
    # Hash algorithm used for content-addressed storage + dedup
    hash_algorithm: Literal["sha256", "blake3"] = "sha256"

    # Hash-prefix directory levels: levels=2 => files/aa/bb/<hash>.ext
    hash_levels: int = 2

    # Streaming chunk size when reading files for hashing (bytes)
    chunk_size: int = 1 << 20  # 1 MiB

    # Hard upper limit on a single file's size (bytes). 0 = unlimited.
    max_bytes: int = 500 * 1024 * 1024  # 500 MiB

    # MIME allowlist. Files whose mime type does not start with ANY
    # of these prefixes are rejected.
    allowed_mime_prefixes: list[str] = Field(
        default_factory=lambda: [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument",
            "application/msword",
            "application/vnd.ms-",
            "text/",
            "image/",
        ]
    )
