---
name: styx-ingest
description: "Archive a document file (PDF, DOCX, XLSX, Markdown, plain text) into Styx via styx_ingest_document. Use when: (1) the user attached or pointed to a file on disk and asked you to read / use / index it, (2) a long external source needs to be searchable later through styx_search_archive, (3) you want chunks of a document to live alongside chunks of subjective memory for hybrid retrieval. NOT for: short pasted text that fits a single styx_store (≤2400 chars subjective material) or styx_ingest_experience (≤2400 chars pipeline payload). The document body does NOT appear in styx_recall — it lives in the archive and is queried through styx_search_archive. A successful first ingest also creates one short archival act-marker memory with memory_domain=external_evidence and line_eligible=false; deduplicated retries do not create another marker. The original file stays on disk; Styx stores parsed text, chunks, embeddings, and that bounded marker."
---

# Styx Ingest (file → archive)

Save a document file into the Styx archive. In the Styx data model this is
**external evidence**, not a subjective trace: its chunks become searchable
through `styx_search_archive`, but storage alone does not make them
`line_eligible`. This is an engineering boundary, not a claim about whether
the surrounding system is a person or conscious.

## Why this is separate from `styx_store` and `styx_ingest_experience`

| Tool | Channel | Effect on `recall` | Limit | Typical source |
|---|---|---|---|---|
| `styx_store` | subjective write | YES — inject in recall | ≤2400 chars (auto-routes longer) | LLM crystallised a fragment of trajectory |
| `styx_ingest_experience` | external-evidence write | NO subjective recall merely by storage | ≤2400 chars | Telegram/email/sensor pipeline |
| `styx_ingest_document` | **file archive** | **Body: NO — pull-only via search_archive; one external-evidence act marker is non-eligible** | up to ingest_doc_max_bytes (default 50 MiB) | User attached a PDF/DOCX/XLSX/MD/TXT |

The current conceptual sources are [IAmBook][iambook] and its applied
continuation [IAmPhilosophyOfSilicon][silicon]. Styx implements their boundary
with explicit domains: diary records what passed through, archive stores cited
external material, and only validated subjective traces may enter the live
line. These tables and gates are one implementation policy, not ontology.

[iambook]: https://github.com/colibri11/IAm/blob/main/IAmBook.md
[silicon]: https://github.com/colibri11/IAm/blob/main/IAmPhilosophyOfSilicon.md

## When to call `styx_ingest_document`

Call it when:

- The user shares a file path (PDF, Word, Excel, Markdown, plain text) and expects you to read / use / index its content.
- Content is bulky (multi-page) — `styx_store` would auto-route long content too but always as subjective tail; `styx_ingest_document` is the explicit archive channel without subjective semantics.
- You expect to need the same document chunks again later in this or future sessions via `styx_search_archive`.

Do NOT call it for:

- Code that you can re-read with file tools — the agent's read_file is sufficient for "I need to see this now".
- Short pasted text (use `styx_store` if it is subjective, `styx_ingest_experience` if pipeline).
- Anything the user explicitly said NOT to archive (privacy / scratch).

## Supported formats

| Extension | Format | Notes |
|---|---|---|
| `.pdf` | PDF | Text layer only. Image-only PDFs (scans) return `422 empty document` — no OCR in this wave. Encrypted PDFs not supported. |
| `.docx` | Word | Paragraphs + table rows. Headers/footers skipped. |
| `.xlsx` | Excel | Each sheet → block. Formulas as cached values (`data_only=True`). |
| `.md`, `.markdown` | Markdown | Raw markdown preserved (not rendered to plain text). |
| `.txt`, `.text` | Plain text | UTF-8 with `errors='replace'` fallback. |

Anything else (PPTX / RTF / ODT / CSV / image) → `422 unsupported extension`.

## How to call it

```
styx_ingest_document({
  path: "/var/lib/styx/uploads/spec-v2.pdf",
  source_ref: "support-ticket-#142",          // optional, free-form
  visibility: "private",                       // optional cosmetic label
  metadata: { uploaded_by: "agent_demo", topic: "ingestion-spec" }
})
```

### Path requirements

- **Absolute path** — relative paths return `422`.
- Inside `STYX_INGEST_DOC_ROOTS` whitelist if the deployment configured one. Production deployments usually restrict to specific user directories (e.g. `/var/lib/styx/docs`).
- File must exist and be a regular file (not a directory, not a special file).
- File size must be ≤ `ingest_doc_max_bytes` (default 50 MiB).

### Idempotency

The same file ingested twice returns:

```
{
  "document_id": "<existing-uuid>",
  "deduplicated": true,
  "chunks_count": 0
}
```

Core hashes the file bytes (SHA256) and looks up `(agent_id, content_hash)` against a partial UNIQUE index. No re-parsing, no re-embedding, no duplicate chunks. Safe to retry.

If the same content arrives under a different filename, deduplication still applies — the hash is from bytes, not the path. If the file content changes, you get a new document.

You can override the hash with an explicit `content_hash` if you know two different files should be treated as the same (rare).

## What happens server-side

1. Path is validated (absolute, within whitelist, size ≤ limit, exists, regular file).
2. SHA256 is computed.
3. If hash matches an existing document for this agent, return `deduplicated=true` and that document's id.
4. Magic bytes are verified against the extension (PDF starts with `%PDF-`, DOCX/XLSX with PK zip signature). Mismatch → `422`.
5. Parser runs (pypdf / python-docx / openpyxl / builtin).
6. Empty text after parsing → `422 empty document` (e.g. image-only PDF).
7. Text is chunked (≤1600 chars per chunk with 320-char overlap) and each chunk is embedded.
8. One `documents` row is INSERT'ed with `file_path`, `original_name`, `mime_type`, `size_bytes`, `content_hash`, and parser metadata (`page_count` / `sheet_names` / `paragraph_count` / `line_count`).
9. Chunks are INSERT'ed in one batch with their embeddings.
10. Exactly one short act-marker `memories` row is created for a successful
    first ingest. It is `memory_domain=external_evidence` with
    `line_eligible=false`, so it records that archival ingestion occurred but
    cannot enter subjective recall/will. The document body remains only in
    `documents` + `chunks`. A deduplicated retry creates no second marker.

## How to retrieve archived content

After ingestion, the document lives in `documents` + `chunks`. Use:

```
styx_search_archive({
  query: "ingestion spec idempotency",
  limit: 10,
  scope: "documents"   // or "all" for documents+dialogue
})
```

`styx_recall` will **not** return chunks or body text from ingested documents —
that is by design. The archival act marker is external evidence and
`line_eligible=false`, not a subjective summary of the document. If the user
asks "what was in the spec?", search the archive; if they ask "what did we
decide about X?" — that is recall territory (subjective memory of decision).

## Errors you may see

| Status | Detail substring | Cause | Action |
|---|---|---|---|
| 422 | `must be absolute` | Relative path | Provide absolute path |
| 422 | `file not found` | Path does not exist | Verify user file path |
| 422 | `not a regular file` | Path is a dir / device | Provide file path, not directory |
| 422 | `outside allowed roots` | Whitelist enforcement | Path must be under one of `STYX_INGEST_DOC_ROOTS` |
| 422 | `file too large` | Exceeds `ingest_doc_max_bytes` | Split or skip |
| 422 | `unsupported extension` | Not PDF/DOCX/XLSX/MD/TXT | Convert externally first |
| 422 | `mime mismatch` | File header doesn't match extension | Likely renamed file; rename back or skip |
| 422 | `encrypted document` | Password-protected PDF | Ask user to decrypt; password ingest not supported |
| 422 | `empty document` | No extractable text (image-only PDF / blank file) | OCR not in scope — ask user for text version |
| 503 | `disabled` | `STYX_INGEST_DOC_ENABLED=0` | Deployment switched the channel off |

Report errors to the user honestly; do not retry with different paths unless you have new information. Do not call `styx_ingest_document` repeatedly on the same path expecting different behaviour — it is deterministic.
