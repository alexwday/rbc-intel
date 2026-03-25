# Storage Table Contract

This document defines the Stage 6 storage contract for the ingestion
pipeline.

It applies to:
- Master CSVs written under `storage/masters/`
- Run-local CSVs written under `processing/storage/`
- PostgreSQL tables refreshed from those CSVs

Use this as the reference for retrieval queries, storage validation, and
research/report data planning.

## Scope

- The storage stage owns the columns listed below.
- CSV headers are the canonical contract for storage-owned fields.
- PostgreSQL sync is a full refresh of these tables from the CSV masters.
- The live database may contain additional legacy columns not listed here.
  Those columns are out of scope for the storage contract.
- All vector columns use `VECTOR(3072)`.
- Logical relationships are stable, but foreign keys are not currently
  enforced in the database DDL.

## Common Conventions

- `file_path` is the absolute path to the source document and is the join
  anchor between catalog and document-level storage.
- `document_id` is a deterministic SHA-256 identifier derived from
  `file_path`.
- Section-, subsection-, chunk-, metric-, keyword-, dense-table-, and
  sheet-level IDs are deterministic SHA-256 identifiers derived from stable
  document coordinates.
- Empty optional text values are stored as empty strings.
- Missing vectors are stored as SQL `NULL`.
- JSON columns store pipeline payload fragments as `JSONB`.

## Join Map

These are logical relationships used by retrieval and downstream services:

- `document_catalog.file_path = documents.file_path`
- `documents.document_id = document_sections.document_id`
- `documents.document_id = document_subsections.document_id`
- `documents.document_id = document_chunks.document_id`
- `documents.document_id = document_dense_tables.document_id`
- `documents.document_id = document_keywords.document_id`
- `documents.document_id = document_metrics.document_id`
- `documents.document_id = document_sheet_summaries.document_id`
- `documents.document_id = document_sheet_context_chains.document_id`
- `document_sections.section_id = document_subsections.section_id`

## Table Overview

| Table | Grain | Primary key | Vector columns |
| --- | --- | --- | --- |
| `document_catalog` | One row per current source file | `file_path` | None |
| `documents` | One row per finalized document | `document_id` | `summary_embedding` |
| `document_sections` | One row per primary section | `section_id` | None |
| `document_subsections` | One row per subsection | `subsection_id` | None |
| `document_chunks` | One row per retrieval chunk | `chunk_id` | `embedding` |
| `document_dense_tables` | One row per dense-table region | `dense_table_id` | None |
| `document_keywords` | One row per embedded keyword | `keyword_id` | `embedding` |
| `document_metrics` | One row per embedded workbook metric | `metric_id` | `embedding` |
| `document_sheet_summaries` | One row per workbook sheet summary | `sheet_summary_id` | None |
| `document_sheet_context_chains` | One row per sheet continuation chain | `chain_id` | None |

## `document_catalog`

Grain: one row per file currently present under the configured input base.

Primary key: `file_path`

Columns:
- `data_source TEXT NOT NULL` — top-level source folder
- `filter_1 TEXT NOT NULL` — first subfolder filter
- `filter_2 TEXT NOT NULL` — second subfolder filter
- `filter_3 TEXT NOT NULL` — flattened remaining subfolder path
- `filename TEXT NOT NULL` — source filename
- `filetype TEXT NOT NULL` — normalized extension
- `file_size BIGINT NOT NULL` — current file size in bytes
- `date_last_modified DOUBLE PRECISION NOT NULL` — source mtime as Unix timestamp
- `file_hash TEXT NOT NULL` — SHA-256 of source file bytes
- `file_path TEXT PRIMARY KEY` — absolute source path

## `documents`

Grain: one row per finalized document.

Primary key: `document_id`

Logical joins:
- `file_path -> document_catalog.file_path`

Columns:
- `document_id TEXT PRIMARY KEY` — deterministic document identifier
- `file_path TEXT NOT NULL UNIQUE` — absolute source path
- `file_name TEXT NOT NULL` — filename from finalized output
- `filetype TEXT NOT NULL` — normalized extension
- `data_source TEXT NOT NULL` — copied from catalog
- `filter_1 TEXT NOT NULL` — copied from catalog
- `filter_2 TEXT NOT NULL` — copied from catalog
- `filter_3 TEXT NOT NULL` — copied from catalog
- `file_size BIGINT NOT NULL` — copied from catalog
- `date_last_modified DOUBLE PRECISION NOT NULL` — copied from catalog
- `file_hash TEXT NOT NULL` — copied from catalog
- `title TEXT NOT NULL` — finalized document title
- `publication_date TEXT NOT NULL` — finalized publication date text
- `authors_text TEXT NOT NULL` — semicolon-delimited author string
- `document_type TEXT NOT NULL` — finalized document type
- `abstract TEXT NOT NULL` — finalized abstract
- `metadata_json JSONB NOT NULL` — full `document_metadata` object
- `document_summary TEXT NOT NULL` — top-level summary
- `document_description TEXT NOT NULL` — retrieval description
- `document_usage TEXT NOT NULL` — intended usage guidance
- `structure_type TEXT NOT NULL` — finalized structure label
- `structure_confidence TEXT NOT NULL` — confidence label
- `degradation_signals_json JSONB NOT NULL` — list of degradation flags
- `summary_embedding VECTOR(3072)` — document-level semantic embedding
- `page_count INTEGER NOT NULL` — total page/sheet/slide count
- `primary_section_count INTEGER NOT NULL` — number of primary sections
- `subsection_count INTEGER NOT NULL` — number of subsections
- `chunk_count INTEGER NOT NULL` — number of stored chunks
- `dense_table_count INTEGER NOT NULL` — number of dense-table regions
- `extraction_metadata_json JSONB NOT NULL` — extraction/finalization metadata

## `document_sections`

Grain: one row per primary section.

Primary key: `section_id`

Logical joins:
- `document_id -> documents.document_id`

Columns:
- `section_id TEXT PRIMARY KEY` — deterministic section identifier
- `document_id TEXT NOT NULL` — parent document
- `file_path TEXT NOT NULL` — denormalized join path
- `section_number INTEGER NOT NULL` — section ordinal within document
- `title TEXT NOT NULL` — section title
- `page_start INTEGER NOT NULL` — first page in section
- `page_end INTEGER NOT NULL` — last page in section
- `page_count INTEGER NOT NULL` — section page count
- `overview TEXT NOT NULL` — section overview summary
- `key_topics_json JSONB NOT NULL` — list of key topics
- `key_metrics_json JSONB NOT NULL` — key metric name/value map
- `key_findings_json JSONB NOT NULL` — list of findings
- `notable_facts_json JSONB NOT NULL` — list of facts
- `is_fallback BOOLEAN NOT NULL` — summary was fallback-generated
- `summary_json JSONB NOT NULL` — full section summary object

## `document_subsections`

Grain: one row per subsection.

Primary key: `subsection_id`

Logical joins:
- `document_id -> documents.document_id`
- `section_id -> document_sections.section_id`

Columns:
- `subsection_id TEXT PRIMARY KEY` — deterministic subsection identifier
- `section_id TEXT NOT NULL` — parent section
- `document_id TEXT NOT NULL` — parent document
- `file_path TEXT NOT NULL` — denormalized join path
- `section_number INTEGER NOT NULL` — parent section ordinal
- `subsection_number INTEGER NOT NULL` — subsection ordinal within section
- `title TEXT NOT NULL` — subsection title
- `page_start INTEGER NOT NULL` — first page in subsection
- `page_end INTEGER NOT NULL` — last page in subsection
- `page_count INTEGER NOT NULL` — subsection page count
- `overview TEXT NOT NULL` — subsection overview
- `key_topics_json JSONB NOT NULL` — list of key topics
- `key_metrics_json JSONB NOT NULL` — key metric name/value map
- `key_findings_json JSONB NOT NULL` — list of findings
- `notable_facts_json JSONB NOT NULL` — list of facts
- `is_fallback BOOLEAN NOT NULL` — summary was fallback-generated
- `summary_json JSONB NOT NULL` — full subsection summary object

## `document_chunks`

Grain: one row per retrieval chunk.

Primary key: `chunk_id`

Logical joins:
- `document_id -> documents.document_id`

Columns:
- `chunk_id TEXT PRIMARY KEY` — deterministic chunk identifier
- `document_id TEXT NOT NULL` — parent document
- `file_path TEXT NOT NULL` — denormalized join path
- `chunk_number INTEGER NOT NULL` — chunk ordinal within document
- `page_number INTEGER NOT NULL` — source page/slide/sheet number
- `content TEXT NOT NULL` — stored retrieval text
- `primary_section_number INTEGER NOT NULL` — parent section ordinal
- `primary_section_name TEXT NOT NULL` — parent section title
- `subsection_number INTEGER NOT NULL` — parent subsection ordinal or `0`
- `subsection_name TEXT NOT NULL` — parent subsection title or empty string
- `hierarchy_path TEXT NOT NULL` — human-readable section path
- `primary_section_page_count INTEGER NOT NULL` — section page count
- `subsection_page_count INTEGER NOT NULL` — subsection page count
- `embedding_prefix TEXT NOT NULL` — LLM-generated summary prefix
- `embedding VECTOR(3072)` — chunk semantic embedding
- `is_dense_table_description BOOLEAN NOT NULL` — chunk stands in for a dense table
- `dense_table_routing_json JSONB NOT NULL` — dense-table routing metadata
- `metadata_json JSONB NOT NULL` — chunk metadata payload

## `document_dense_tables`

Grain: one row per dense-table region replaced in chunked content.

Primary key: `dense_table_id`

Logical joins:
- `document_id -> documents.document_id`

Columns:
- `dense_table_id TEXT PRIMARY KEY` — deterministic region identifier
- `document_id TEXT NOT NULL` — parent document
- `file_path TEXT NOT NULL` — denormalized join path
- `region_id TEXT NOT NULL` — region identifier from content prep
- `used_range TEXT NOT NULL` — sheet cell range
- `sheet_name TEXT NOT NULL` — workbook sheet name
- `page_title TEXT NOT NULL` — page-like title when available
- `description_generation_mode TEXT NOT NULL` — one-shot, batched, deterministic, or empty
- `replacement_content TEXT NOT NULL` — text inserted into chunk stream
- `routing_metadata_json JSONB NOT NULL` — routing metadata used for chunking
- `dense_table_description_json JSONB NOT NULL` — final dense-table description object
- `dense_table_eda_json JSONB NOT NULL` — EDA payload for the region
- `raw_content_json JSONB NOT NULL` — raw serialized region rows

## `document_keywords`

Grain: one row per keyword embedding.

Primary key: `keyword_id`

Logical joins:
- `document_id -> documents.document_id`

Columns:
- `keyword_id TEXT PRIMARY KEY` — deterministic keyword identifier
- `document_id TEXT NOT NULL` — parent document
- `file_path TEXT NOT NULL` — denormalized join path
- `keyword TEXT NOT NULL` — keyword text
- `page_number INTEGER NOT NULL` — source page/slide/sheet number
- `page_title TEXT NOT NULL` — source page title when available
- `section TEXT NOT NULL` — source section label
- `embedding VECTOR(3072)` — keyword embedding

## `document_metrics`

Grain: one row per extracted workbook metric.

Primary key: `metric_id`

Logical joins:
- `document_id -> documents.document_id`

Columns:
- `metric_id TEXT PRIMARY KEY` — deterministic metric identifier
- `document_id TEXT NOT NULL` — parent document
- `file_path TEXT NOT NULL` — denormalized join path
- `metric_name TEXT NOT NULL` — metric label
- `page_number INTEGER NOT NULL` — source page/sheet number
- `sheet_name TEXT NOT NULL` — workbook sheet name
- `region_id TEXT NOT NULL` — region identifier
- `used_range TEXT NOT NULL` — sheet cell range
- `platform TEXT NOT NULL` — platform/domain classification
- `sub_platform TEXT NOT NULL` — sub-platform classification
- `periods_available_json JSONB NOT NULL` — detected periods list
- `embedding VECTOR(3072)` — metric embedding

## `document_sheet_summaries`

Grain: one row per workbook sheet summary.

Primary key: `sheet_summary_id`

Logical joins:
- `document_id -> documents.document_id`

Columns:
- `sheet_summary_id TEXT PRIMARY KEY` — deterministic sheet summary identifier
- `document_id TEXT NOT NULL` — parent document
- `file_path TEXT NOT NULL` — denormalized join path
- `sheet_name TEXT NOT NULL` — workbook sheet name
- `handling_mode TEXT NOT NULL` — `page_like` or `dense_table_candidate`
- `summary TEXT NOT NULL` — sheet-level summary
- `usage TEXT NOT NULL` — sheet-level usage guidance

## `document_sheet_context_chains`

Grain: one row per workbook continuation-chain entry.

Primary key: `chain_id`

Logical joins:
- `document_id -> documents.document_id`

Columns:
- `chain_id TEXT PRIMARY KEY` — deterministic chain identifier
- `document_id TEXT NOT NULL` — parent document
- `file_path TEXT NOT NULL` — denormalized join path
- `sheet_index INTEGER NOT NULL` — workbook sheet ordinal
- `sheet_name TEXT NOT NULL` — workbook sheet name
- `context_sheet_indices_json JSONB NOT NULL` — prior related sheet indices

## Operational Notes

- The storage stage can be run independently with:

```bash
.venv/bin/python -m ingestion.main --storage-only
```

- To push the current master CSVs into PostgreSQL during that run:

```bash
STORAGE_PUSH_TO_POSTGRES=true .venv/bin/python -m ingestion.main --storage-only
```

- `document_catalog` reflects the current input folder base on every run.
- The other tables reflect the current source snapshot plus any unresolved
  finalized outputs recovered from archived runs.
- If a supported current file cannot be resolved to a finalized output, the
  storage stage fails rather than writing a partial snapshot.
