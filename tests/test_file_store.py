"""FileStore tests: hash-based storage, dedup, metadata."""

from __future__ import annotations

import pytest

from config import FilesConfig, RelationalConfig, SQLiteConfig
from parser.blob_store import LocalBlobStore, LocalStoreConfig, file_key
from persistence.files import FileStore
from persistence.store import Store


@pytest.fixture
def env(tmp_path):
    rel = Store(
        RelationalConfig(
            backend="sqlite",
            sqlite=SQLiteConfig(path=str(tmp_path / "hi.db")),
        )
    )
    rel.connect()
    rel.ensure_schema()

    blob = LocalBlobStore(
        LocalStoreConfig(
            root=str(tmp_path / "blobs"),
            public_base_url="http://host/static",
        )
    )
    files = FileStore(FilesConfig(), blob, rel)
    yield rel, blob, files
    rel.close()


# ---------------------------------------------------------------------------


class TestFileKey:
    def test_two_levels(self):
        h = "aabbccddeeff" + "0" * 52  # 64 chars
        assert file_key(h, "pdf", levels=2) == f"files/aa/bb/{h}.pdf"

    def test_one_level(self):
        h = "aabbccdd" + "0" * 56
        assert file_key(h, "pdf", levels=1) == f"files/aa/{h}.pdf"


# ---------------------------------------------------------------------------


class TestStoreBytes:
    def test_basic_upload(self, env):
        _rel, blob, files = env
        record = files.store(
            b"%PDF-1.4\nhello world\n%%EOF\n",
            original_name="report.pdf",
            mime_type="application/pdf",
        )
        assert record["file_id"]
        assert len(record["content_hash"]) == 64  # sha256 hex
        assert record["storage_key"].startswith("files/")
        assert record["storage_key"].endswith(".pdf")
        assert record["original_name"] == "report.pdf"
        assert "_" in record["display_name"]
        assert record["display_name"].endswith(".pdf")
        assert record["size_bytes"] == len(b"%PDF-1.4\nhello world\n%%EOF\n")
        assert blob.exists(record["storage_key"])

    def test_dedup_same_content_shares_blob(self, env):
        _rel, _blob, files = env
        data = b"%PDF-1.4\nsame content\n"
        r1 = files.store(data, original_name="a.pdf", mime_type="application/pdf")
        r2 = files.store(data, original_name="b.pdf", mime_type="application/pdf")

        assert r1["content_hash"] == r2["content_hash"]
        assert r1["storage_key"] == r2["storage_key"]  # same blob
        assert r1["file_id"] != r2["file_id"]  # different rows
        assert r1["original_name"] != r2["original_name"]

    def test_different_content_different_hash(self, env):
        _rel, _blob, files = env
        r1 = files.store(b"%PDF-1.4 file A", original_name="a.pdf", mime_type="application/pdf")
        r2 = files.store(b"%PDF-1.4 file B", original_name="a.pdf", mime_type="application/pdf")
        assert r1["content_hash"] != r2["content_hash"]
        assert r1["storage_key"] != r2["storage_key"]


class TestMimeValidation:
    def test_rejects_disallowed_mime(self, env):
        _rel, _blob, files = env
        with pytest.raises(ValueError, match="not allowed"):
            files.store(
                b"dangerous",
                original_name="x.exe",
                mime_type="application/x-msdownload",
            )

    def test_empty_file_rejected(self, env):
        _rel, _blob, files = env
        with pytest.raises(ValueError, match="empty"):
            files.store(b"", original_name="empty.pdf", mime_type="application/pdf")

    def test_max_bytes_enforced(self, tmp_path):
        rel = Store(
            RelationalConfig(
                backend="sqlite",
                sqlite=SQLiteConfig(path=str(tmp_path / "x.db")),
            )
        )
        rel.connect()
        rel.ensure_schema()
        blob = LocalBlobStore(LocalStoreConfig(root=str(tmp_path / "b")))
        cfg = FilesConfig(max_bytes=10)
        fs = FileStore(cfg, blob, rel)
        with pytest.raises(ValueError, match="too large"):
            fs.store(
                b"x" * 100,
                original_name="big.pdf",
                mime_type="application/pdf",
            )
        rel.close()


class TestPathSource:
    def test_store_from_path(self, env, tmp_path):
        _rel, blob, files = env
        src = tmp_path / "upload.pdf"
        src.write_bytes(b"%PDF-1.4\ndisk source\n")
        record = files.store(src, original_name="upload.pdf", mime_type="application/pdf")
        assert record["size_bytes"] == src.stat().st_size
        assert blob.exists(record["storage_key"])

    def test_materialize_reads_back_file(self, env, tmp_path):
        _rel, _blob, files = env
        raw = b"%PDF-1.4\noriginal bytes\n"
        record = files.store(raw, original_name="x.pdf", mime_type="application/pdf")
        out = tmp_path / "out.pdf"
        files.materialize(record["file_id"], out)
        assert out.read_bytes() == raw


class TestUrlFor:
    def test_url_for_returns_public_url(self, env):
        _rel, _blob, files = env
        record = files.store(
            b"%PDF-1.4\n",
            original_name="x.pdf",
            mime_type="application/pdf",
        )
        url = files.url_for(record["file_id"])
        assert url and url.startswith("http://host/static/")
