"""
Storage configuration.

Lives at the top level because BlobStore is used by the parser
(writing figures) AND by the retrieval / citation layers (serving
URLs to the frontend), and in the future possibly by the ingestion
pipeline. Keeping it here avoids circular imports.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from parser.blob_store import (
    LocalStoreConfig,
    OSSStoreConfig,
    S3StoreConfig,
    StorageConfig,
)


class LocalStorageModel(BaseModel):
    root: str = "./storage/figures"
    public_base_url: str | None = None

    def to_dataclass(self) -> LocalStoreConfig:
        return LocalStoreConfig(root=self.root, public_base_url=self.public_base_url)


class S3StorageModel(BaseModel):
    endpoint: str
    bucket: str
    region: str = "us-east-1"
    access_key_env: str = "S3_ACCESS_KEY"
    secret_key_env: str = "S3_SECRET_KEY"
    prefix: str = ""
    public_base_url: str | None = None

    def to_dataclass(self) -> S3StoreConfig:
        return S3StoreConfig(**self.model_dump())


class OSSStorageModel(BaseModel):
    endpoint: str
    bucket: str
    access_key_env: str = "OSS_ACCESS_KEY"
    secret_key_env: str = "OSS_SECRET_KEY"
    prefix: str = ""
    public_base_url: str | None = None

    def to_dataclass(self) -> OSSStoreConfig:
        return OSSStoreConfig(**self.model_dump())


class StorageModel(BaseModel):
    mode: Literal["local", "s3", "oss"] = "local"
    local: LocalStorageModel | None = Field(default_factory=LocalStorageModel)
    s3: S3StorageModel | None = None
    oss: OSSStorageModel | None = None

    @model_validator(mode="after")
    def _check_mode_section(self) -> StorageModel:
        if self.mode == "s3" and self.s3 is None:
            raise ValueError("storage.mode=s3 but storage.s3 section missing")
        if self.mode == "oss" and self.oss is None:
            raise ValueError("storage.mode=oss but storage.oss section missing")
        if self.mode == "local" and self.local is None:
            self.local = LocalStorageModel()
        return self

    def to_dataclass(self) -> StorageConfig:
        return StorageConfig(
            mode=self.mode,
            local=self.local.to_dataclass() if self.local else None,
            s3=self.s3.to_dataclass() if self.s3 else None,
            oss=self.oss.to_dataclass() if self.oss else None,
        )
