"""Config-level tests for the persistence layer (no DB required)."""

from __future__ import annotations

import pytest

from config import PersistenceConfig, RelationalConfig, VectorConfig
from config.persistence import (
    ChromaConfig,
    MySQLConfig,
)


class TestDefaults:
    def test_default_is_postgres_pgvector(self):
        cfg = PersistenceConfig()
        assert cfg.relational.backend == "postgres"
        assert cfg.vector.backend == "pgvector"
        assert cfg.relational.postgres is not None
        assert cfg.vector.pgvector is not None


class TestValidCombinations:
    def test_postgres_chromadb(self):
        cfg = PersistenceConfig(
            relational=RelationalConfig(backend="postgres"),
            vector=VectorConfig(
                backend="chromadb",
                chromadb=ChromaConfig(persist_directory="/tmp/x"),
            ),
        )
        assert cfg.vector.backend == "chromadb"

    def test_mysql_chromadb(self):
        cfg = PersistenceConfig(
            relational=RelationalConfig(
                backend="mysql",
                mysql=MySQLConfig(),
            ),
            vector=VectorConfig(
                backend="chromadb",
                chromadb=ChromaConfig(persist_directory="/tmp/x"),
            ),
        )
        assert cfg.relational.backend == "mysql"
        assert cfg.vector.backend == "chromadb"


class TestInvalidCombinations:
    def test_mysql_pgvector_rejected(self):
        with pytest.raises(ValueError, match="mysql.*pgvector"):
            PersistenceConfig(
                relational=RelationalConfig(
                    backend="mysql",
                    mysql=MySQLConfig(),
                ),
                vector=VectorConfig(backend="pgvector"),
            )

    def test_mysql_without_section_rejected(self):
        with pytest.raises(ValueError, match="mysql section missing"):
            RelationalConfig(backend="mysql", mysql=None)

    def test_chromadb_without_section_rejected(self):
        with pytest.raises(ValueError, match="chromadb section missing"):
            VectorConfig(backend="chromadb", chromadb=None)
