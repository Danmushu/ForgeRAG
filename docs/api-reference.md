# API Reference

ForgeRAG exposes a REST API at `/api/v1/`. Interactive documentation is available at:

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

All request/response bodies use JSON. File uploads use `multipart/form-data`.

---

## Query

### Ask a Question

```
POST /api/v1/query
```

Ask a question and get an answer with citations.

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | yes | The question to ask |
| `conversation_id` | string | no | Continue an existing conversation (multi-turn) |
| `stream` | bool | no | `true` for Server-Sent Events streaming (default: `false`) |
| `filter` | object | no | Metadata filter (e.g., `{"doc_id": "..."}`) |

**Normal response** (`stream: false`):

```json
{
  "answer": "The answer text with citations...",
  "citations": [
    {
      "citation_id": "c_1",
      "chunk_id": "chunk_abc123",
      "doc_id": "doc_xyz",
      "page_no": 5,
      "bbox": {"x0": 72, "y0": 200, "x1": 540, "y1": 280},
      "snippet": "Relevant text excerpt...",
      "file_id": "file_001"
    }
  ],
  "conversation_id": "conv_123",
  "stats": {
    "retrieval_ms": 450,
    "generation_ms": 1200,
    "vector_hits": 15,
    "tree_hits": 8,
    "bm25_hits": 12,
    "merged_chunks": 10
  }
}
```

**Streaming response** (`stream: true`):

Returns `text/event-stream` with events:

| Event | Data | Description |
|-------|------|-------------|
| `progress` | `{"phase": "query_understanding"}` | Current retrieval phase |
| `retrieval` | `{"citations": [...], "stats": {...}}` | Retrieval results |
| `delta` | `{"text": "token"}` | Generated text token |
| `done` | `{"answer": "...", "citations": [...]}` | Final complete response |

**Example (streaming with curl):**

```bash
curl -N -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the revenue for Q3?", "stream": true}'
```

---

## Documents

### Upload and Ingest

```
POST /api/v1/documents
```

Upload a file and queue it for ingestion.

**Request:** `multipart/form-data` with `file` field.

**Response** (202 Accepted):

```json
{
  "doc_id": "doc_abc123",
  "file_id": "file_xyz",
  "status": "pending"
}
```

The document is processed asynchronously. Poll the document detail endpoint to check status.

### Upload and Ingest (multipart)

```
POST /api/v1/documents/upload-and-ingest
```

Same as above, alternative endpoint name.

### List Documents

```
GET /api/v1/documents?limit=50&offset=0&search=quarterly&status=ready
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 50 | Page size |
| `offset` | int | 0 | Offset |
| `search` | string | — | Search in filenames |
| `status` | string | — | Filter by status: `pending`, `parsing`, `ready`, `error` |

**Response:**

```json
{
  "items": [
    {
      "doc_id": "doc_abc123",
      "file_name": "annual_report.pdf",
      "format": "pdf",
      "status": "ready",
      "embed_status": "done",
      "enrich_status": "done",
      "num_chunks": 142,
      "num_blocks": 580,
      "file_size_bytes": 2457600,
      "created_at": "2025-01-15T10:30:00Z"
    }
  ],
  "total": 1
}
```

### Get Document Detail

```
GET /api/v1/documents/{doc_id}
```

Returns full document metadata including processing status, timing, and statistics.

### Get Document Blocks

```
GET /api/v1/documents/{doc_id}/blocks?limit=100&offset=0
```

Returns parsed blocks with page numbers, bounding boxes, and types.

### Get Document Chunks

```
GET /api/v1/documents/{doc_id}/chunks?limit=50&offset=0
```

Returns chunks with content, token counts, section paths, and block IDs.

### Get Document Tree

```
GET /api/v1/documents/{doc_id}/tree
```

Returns the full hierarchical tree structure:

```json
{
  "root_id": "node_root",
  "generation_method": "toc",
  "quality_score": 0.92,
  "nodes": {
    "node_root": {
      "node_id": "node_root",
      "title": "Document",
      "level": 0,
      "page_start": 1,
      "page_end": 50,
      "children": ["node_ch1", "node_ch2"],
      "block_ids": [],
      "summary": null
    },
    "node_ch1": {
      "node_id": "node_ch1",
      "title": "Chapter 1: Introduction",
      "level": 1,
      "page_start": 1,
      "page_end": 12,
      "children": ["node_s1_1", "node_s1_2"],
      "block_ids": ["blk_001", "blk_002"],
      "summary": "Overview of the project..."
    }
  }
}
```

### Delete Document

```
DELETE /api/v1/documents/{doc_id}
```

Soft-deletes the document (marks status).

### Reparse Document

```
POST /api/v1/documents/{doc_id}/reparse
```

Force re-ingestion with a new parse version.

---

## Files

### Upload File

```
POST /api/v1/files
```

Upload a file without triggering ingestion. Useful for two-step workflows.

**Request:** `multipart/form-data` with `file` field.

### Upload from URL

```
POST /api/v1/files/from-url
```

Fetch a file from a URL (SSRF-protected).

**Request body:**

```json
{
  "url": "https://example.com/report.pdf",
  "original_name": "report.pdf"
}
```

### Download File

```
GET /api/v1/files/{file_id}/download
```

Returns the file with `Content-Disposition: attachment`.

### Preview File

```
GET /api/v1/files/{file_id}/preview
```

Returns the file with `Content-Disposition: inline` (for PDF viewer embedding).

### List Files

```
GET /api/v1/files?limit=50&offset=0
```

### Delete File

```
DELETE /api/v1/files/{file_id}
```

---

## Chunks

### Get Chunk by ID

```
GET /api/v1/chunks/{chunk_id}
```

Returns a single chunk with full metadata.

### Get Block Image

```
GET /api/v1/blocks/{block_id}/image
```

Returns the extracted image for a figure block.

---

## Conversations

### List Conversations

```
GET /api/v1/conversations?limit=20&offset=0
```

### Get Conversation

```
GET /api/v1/conversations/{conversation_id}
```

Returns all turns (questions + answers + citations) in the conversation.

### Delete Conversation

```
DELETE /api/v1/conversations/{conversation_id}
```

---

## Knowledge Graph

### Get Full Graph

```
GET /api/v1/graph?limit=1000
```

Returns all entities and relations for visualization.

### Get Entity Detail

```
GET /api/v1/graph/entities/{entity_id}
```

Returns entity info + neighboring entities + relations.

### Search Entities

```
GET /api/v1/graph/search?q=revenue&top_k=10
```

Fuzzy/substring search over entity names.

### Get Subgraph

```
GET /api/v1/graph/subgraph?entity_ids=e1,e2,e3
```

Returns the subgraph connecting the specified entities.

---

## Settings

### Get All Settings

```
GET /api/v1/settings
```

Returns all settings grouped by category.

### Get Settings by Group

```
GET /api/v1/settings/{group}
```

### Get Single Setting

```
GET /api/v1/settings/key/{key}
```

Key uses dotted notation: `retrieval.vector.top_k`.

### Update Setting

```
PUT /api/v1/settings/key/{key}
```

**Request body:**

```json
{
  "value": 50
}
```

Changes take effect immediately — no restart required.

### Batch Update Settings

```
PUT /api/v1/settings
```

**Request body:**

```json
{
  "settings": {
    "retrieval.vector.top_k": 50,
    "answering.generator.temperature": 0.2
  }
}
```

### Reset Setting to Default

```
DELETE /api/v1/settings/{key}
```

Removes the DB override; reverts to YAML config value.

### Apply All Overrides

```
POST /api/v1/settings/apply
```

Re-applies all DB overrides to the running config. Useful after direct DB edits.

---

## LLM Providers

### List Providers

```
GET /api/v1/llm-providers
```

Returns configured LLM providers.

### Add Provider

```
POST /api/v1/llm-providers
```

Register a new LLM provider (OpenAI, Azure, Ollama, etc.).

---

## System

### Health Check

```
GET /api/v1/health
```

Returns `{"status": "ok"}` if the server is running.

### System Info

```
GET /api/v1/system/info
```

Returns backend versions, document count, storage usage, and configuration summary.

---

## Traces

### List Retrieval Traces

```
GET /api/v1/traces?limit=20&offset=0
```

Returns retrieval trace history with timing, phases, and LLM call details.

---

## Benchmark

### Start Benchmark

```
POST /api/v1/benchmark/start
```

**Request body:**

```json
{
  "num_questions": 20
}
```

### Get Benchmark Status

```
GET /api/v1/benchmark/status
```

Returns current phase, progress, elapsed time, and estimated remaining time.

### Cancel Benchmark

```
POST /api/v1/benchmark/cancel
```

### Download Benchmark Report

```
GET /api/v1/benchmark/report
```

Returns a JSON report with scores, per-question details, and config snapshot (credentials redacted).

---

## Error Responses

All errors follow a consistent format:

```json
{
  "detail": "Document not found"
}
```

| Status Code | Meaning |
|-------------|---------|
| 400 | Bad request (invalid parameters) |
| 404 | Resource not found |
| 409 | Conflict (e.g., document already ingesting) |
| 413 | File too large |
| 422 | Validation error |
| 500 | Internal server error |
