"""
Smoke test: exercises the full pipeline against real PDFs.

    python scripts/smoke_test.py [--dir tests/pdfs] [--limit 3] [--query "..."]

Steps per PDF:
    1. FileStore.store           (upload + hash + blob)
    2. IngestionPipeline.ingest  (parse + tree + chunk + write)
    3. (optional) embed

Then:
    4. build BM25 index
    5. RetrievalPipeline.retrieve
    6. AnsweringPipeline.ask     (with FakeGenerator, no API key needed)

No network calls. No GPU. No external services. Pure SQLite + local
blob + PyMuPDF + FakeEmbedder + FakeGenerator. The point is to verify
the entire chain runs without crashing on real-world PDFs.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from answering.pipeline import AnsweringPipeline
from answering.prompts import extract_cited_ids
from config import (
    AppConfig,
    FilesConfig,
    LocalStorageModel,
    RelationalConfig,
    SQLiteConfig,
    StorageModel,
)
from ingestion import IngestionPipeline
from parser.blob_store import LocalBlobStore, LocalStoreConfig
from parser.chunker import Chunker
from parser.pipeline import ParserPipeline
from parser.tree_builder import TreeBuilder
from persistence.files import FileStore
from persistence.store import Store
from persistence.vector.base import VectorHit, VectorItem
from retrieval.pipeline import RetrievalPipeline, build_bm25_index

# ---------------------------------------------------------------------------
# Fakes (no network, no GPU)
# ---------------------------------------------------------------------------


class FakeEmbedder:
    backend = "fake"
    dimension = 4
    batch_size = 32

    def embed_texts(self, texts):
        return [[float(len(t) % 7), 0.5, 0.5, float(i % 3)] for i, t in enumerate(texts)]

    def embed_chunks(self, chunks):
        return {
            c.chunk_id: [float(len(c.content) % 7), 0.5, 0.5, float(i % 3)]
            for i, c in enumerate(chunks)
            if c.content.strip()
        }


class FakeVectorStore:
    backend = "fake"
    dimension = 4

    def __init__(self):
        self.items: dict[str, VectorItem] = {}

    def connect(self):
        pass

    def close(self):
        pass

    def ensure_schema(self):
        pass

    def upsert(self, items):
        for it in items:
            self.items[it.chunk_id] = it

    def delete_chunks(self, ids):
        for i in ids:
            self.items.pop(i, None)

    def delete_parse_version(self, doc_id, pv):
        for cid in list(self.items):
            it = self.items[cid]
            if it.doc_id == doc_id and it.parse_version == pv:
                del self.items[cid]

    def search(self, q, *, top_k, filter=None):
        return [
            VectorHit(
                chunk_id=it.chunk_id,
                score=1.0 - i * 0.01,
                doc_id=it.doc_id,
                parse_version=it.parse_version,
                metadata=it.metadata,
            )
            for i, it in enumerate(list(self.items.values())[:top_k])
        ]


class FakeGenerator:
    backend = "fake"
    model = "fake/smoke-test"

    def generate(self, messages):
        user = "\n".join(m["content"] for m in messages if m["role"] == "user")
        # Cite the first marker we see in the prompt
        import re

        markers = re.findall(r"\[c_\d+\]", user)
        cite = markers[0] if markers else ""
        text = f"Smoke test answer based on retrieved context. {cite}"
        return {
            "text": text,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
            "model": self.model,
            "cited_ids": extract_cited_ids(text),
            "latency_ms": 1,
        }


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def build_env(tmp_dir: Path):
    cfg = AppConfig()
    cfg.persistence.relational = RelationalConfig(
        backend="sqlite",
        sqlite=SQLiteConfig(path=str(tmp_dir / "smoke.db")),
    )
    cfg.storage = StorageModel(
        mode="local",
        local=LocalStorageModel(root=str(tmp_dir / "blobs")),
    )
    cfg.files = FilesConfig()
    cfg.embedder.dimension = 4
    cfg.persistence.vector.pgvector.dimension = 4

    rel = Store(cfg.persistence.relational)
    rel.connect()
    rel.ensure_schema()

    blob = LocalBlobStore(LocalStoreConfig(root=str(tmp_dir / "blobs")))
    file_store = FileStore(cfg.files, blob, rel)
    vec = FakeVectorStore()
    emb = FakeEmbedder()

    parser = ParserPipeline.from_config(cfg)
    tree = TreeBuilder(cfg.parser.tree_builder)
    chunker = Chunker(cfg.parser.chunker)

    pipeline = IngestionPipeline(
        file_store=file_store,
        parser=parser,
        tree_builder=tree,
        chunker=chunker,
        relational_store=rel,
        vector_store=vec,
        embedder=emb,
    )
    return cfg, rel, vec, emb, pipeline


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------


def _c(text, color):
    if not sys.stdout.isatty():
        return text
    codes = {"green": "32", "red": "31", "yellow": "33", "dim": "2", "bold": "1", "cyan": "36"}
    return f"\033[{codes.get(color, '0')}m{text}\033[0m"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Smoke test against real PDFs.")
    p.add_argument("--dir", type=Path, default=_ROOT / "tests" / "pdfs")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--query", type=str, default="What is the main contribution of this paper?")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname).1s %(name)s: %(message)s",
    )

    if not args.dir.exists():
        print(f"error: {args.dir} not found", file=sys.stderr)
        return 2

    pdfs = sorted(args.dir.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print("no PDFs found")
        return 2

    print(_c(f"\n{'=' * 60}", "dim"))
    print(_c(f" Smoke test: {len(pdfs)} PDFs", "bold"))
    print(_c(f" Query: {args.query}", "dim"))
    print(_c(f"{'=' * 60}\n", "dim"))

    with tempfile.TemporaryDirectory(prefix="qr_smoke_", ignore_cleanup_errors=True) as tmp:
        tmp_dir = Path(tmp)
        cfg, rel, vec, emb, pipeline = build_env(tmp_dir)

        # --- Phase 1: Ingest ---
        print(_c("Phase 1: Ingest", "cyan"))
        results = []
        for i, pdf in enumerate(pdfs, 1):
            t0 = time.time()
            try:
                r = pipeline.upload_and_ingest(
                    pdf,
                    original_name=pdf.name,
                    mime_type="application/pdf",
                )
                elapsed = int((time.time() - t0) * 1000)
                tag = _c("OK  ", "green")
                detail = f"blocks={r.num_blocks:>3} chunks={r.num_chunks:>3} tree_q={r.tree_quality:.2f}"
                results.append({"ok": True, "result": r})
            except Exception as e:
                elapsed = int((time.time() - t0) * 1000)
                tag = _c("FAIL", "red")
                detail = f"{type(e).__name__}: {e}"
                results.append({"ok": False, "error": str(e)})
                if args.verbose:
                    traceback.print_exc()
            print(f"  [{i:>2}/{len(pdfs)}] {tag} {pdf.name[:40]:<40} {elapsed:>5}ms  {detail}")

        ok_count = sum(1 for r in results if r["ok"])
        fail_count = len(results) - ok_count
        print()
        print(f"  ingested: {_c(str(ok_count), 'green')}  failed: {_c(str(fail_count), 'red')}")

        if ok_count == 0:
            print(_c("\n  no documents ingested, skipping retrieval.\n", "yellow"))
            return 1

        # --- Phase 2: Index stats ---
        print()
        print(_c("Phase 2: Index stats", "cyan"))
        doc_ids = rel.list_document_ids()
        total_chunks = sum(len(rel.get_chunks(d, rel.get_document(d)["active_parse_version"])) for d in doc_ids)
        print(f"  documents:  {len(doc_ids)}")
        print(f"  chunks:     {total_chunks}")
        print(f"  vectors:    {len(vec.items)}")

        # --- Phase 3: BM25 index ---
        print()
        print(_c("Phase 3: Build BM25 index", "cyan"))
        t0 = time.time()
        bm25 = build_bm25_index(rel, cfg.retrieval.bm25)
        print(f"  indexed {len(bm25)} chunks in {int((time.time() - t0) * 1000)}ms")

        # --- Phase 4: Retrieval ---
        print()
        print(_c(f'Phase 4: Retrieval (query: "{args.query[:60]}")', "cyan"))
        retrieval = RetrievalPipeline(
            cfg.retrieval,
            embedder=emb,
            vector_store=vec,
            relational_store=rel,
            bm25_index=bm25,
        )
        t0 = time.time()
        ret_result = retrieval.retrieve(args.query)
        elapsed = int((time.time() - t0) * 1000)
        print(f"  vector_hits:  {ret_result.stats.get('vector_hits', 0)}")
        print(f"  tree_hits:    {ret_result.stats.get('tree_hits', 0)}")
        print(f"  merged:       {ret_result.stats.get('merged_count', 0)}")
        print(f"  reranked:     {ret_result.stats.get('reranked_count', 0)}")
        print(f"  citations:    {len(ret_result.citations)}")
        print(f"  elapsed:      {elapsed}ms")

        if ret_result.citations:
            print()
            print(_c("  Top 5 citations:", "dim"))
            for c in ret_result.citations[:5]:
                hl = c.highlights[0] if c.highlights else None
                page = hl.page_no if hl else "?"
                bbox = f"({hl.bbox[0]:.0f},{hl.bbox[1]:.0f},{hl.bbox[2]:.0f},{hl.bbox[3]:.0f})" if hl else ""
                print(f"    {c.citation_id}  doc={c.doc_id[:20]}  p{page} {bbox}")
                print(f"      snippet: {c.snippet[:80]}...")

        # --- Phase 5: Answer generation ---
        print()
        print(_c("Phase 5: Answer generation (FakeGenerator)", "cyan"))
        answering = AnsweringPipeline(
            cfg.answering,
            retrieval=retrieval,
            generator=FakeGenerator(),
        )
        t0 = time.time()
        answer = answering.ask(args.query)
        elapsed = int((time.time() - t0) * 1000)
        print(f"  answer:       {answer.text[:120]}")
        print(f"  cited:        {[c.citation_id for c in answer.citations_used]}")
        print(f"  finish:       {answer.finish_reason}")
        print(f"  elapsed:      {elapsed}ms")

        # --- Summary ---
        print()
        print(_c(f"{'=' * 60}", "dim"))
        print(_c(" Smoke test complete", "bold"))
        print(f"  PDFs:        {len(pdfs)}")
        print(f"  Ingested:    {ok_count}")
        print(f"  Failed:      {fail_count}")
        print(f"  Chunks:      {total_chunks}")
        print(f"  Citations:   {len(ret_result.citations)}")
        print(f"  Answer len:  {len(answer.text)} chars")
        all_ok = fail_count == 0 and len(ret_result.citations) > 0
        status = _c("PASS", "green") if all_ok else _c("PARTIAL", "yellow")
        print(f"  Status:      {status}")
        print(_c(f"{'=' * 60}\n", "dim"))

        # Close DB before TemporaryDirectory tries to delete it (Windows)
        rel.close()

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
