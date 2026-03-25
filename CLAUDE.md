# Document Ingestion, Research & Report Pipeline

## Project Overview

A Python data processing pipeline that ingests documents from folder-based sources, uses LLM tool calling to extract/classify/summarize/enrich content, and stores structured output in PostgreSQL for retrieval. The system supports three workflows: **Ingestion**, **Research**, and **Report Generation**.

## Tech Stack

- **Language**: Python 3.12+
- **LLM**: OpenAI SDK (tool calling only, no streaming) — architecture is LLM-agnostic; the connector layer must be swappable
- **Database**: PostgreSQL + pgvector
- **Testing**: pytest (target complete coverage)
- **Document parsing**: PyMuPDF, python-docx, python-pptx, openpyxl
- **Embeddings**: pgvector for semantic search

## Architecture

### Ingestion Pipeline

1. **Discovery** — Scans named folders for new/modified/deleted files; subfolders become queryable filters
2. **Extraction** — Per-file, filetype-specific processing. PDF/DOCX/PPTX rendered to PNG via PyMuPDF then extracted via LLM Vision. XLSX parsed structurally via openpyxl. Each processor is a self-contained package under `processors/` with its own prompts, utils, and content preparation logic
3. **Content Preparation** — Passthrough for PDF/DOCX/PPTX. For XLSX dense-table sheets: identifies dense table regions, generates EDA + LLM descriptions, splices descriptions into original sheet content replacing only the dense table rows. Outputs full page-level content (no chunking)
4. **Enrichment** — LLM generates per-page metadata: summary, usage description, section hierarchy, keywords, classifications, entities. Uses processor-specific enrichment prompts
5. **Chunking & Embedding** — (not yet built) Splits enriched content into embeddable chunks, generates pgvector embeddings
6. **Storage** — (not yet built) PostgreSQL + pgvector. Separate tables for document catalog, page content, and extracted tables

### Research Pipeline

1. **Orchestrator Loop** — Conversation with user, clarification, then calls research tool
2. **Data Source Planning** — Selects relevant sources via registry + similarity search
3. **Document Selection** — Enriches filtered catalog with top chunk per file; LLM classifies each as Research/Deep Research/Irrelevant
4. **Research Execution** — Research (metadata + chunks) or Deep Research (full content loop); optional dense table research
5. **Response Synthesis** — Aggregates per-file research into summary; saves outputs to ad-hoc folder

### Report Pipeline

1. **Orchestrator Loop** — Clarifies scope, calls report tool
2. **Research Gap Filling** — Iterative loop: gap analysis, research subagent calls, coverage check (max N iterations)
3. **Report Template** — Creates/updates section structure with per-section writing instructions
4. **Section Generation** — Each section researched and written independently
5. **Final Compilation** — Compiles into DOCX, saves to ad-hoc folder

## Setup Dependencies

### System requirements
- **Python 3.12+**
- **LibreOffice** — headless mode, for DOCX/PPTX to PDF conversion
- **Metrically-compatible fonts** — required so LibreOffice page breaks match Microsoft Word

### Font installation (critical for DOCX/PPTX fidelity)

LibreOffice uses different default fonts than Word. Without metrically-compatible replacements, lines wrap at different positions and page breaks shift — causing tables to split across pages. Install these:

**macOS (Homebrew):**
```bash
brew install --cask font-carlito font-caladea font-liberation
```

**Debian / Ubuntu:**
```bash
apt-get install fonts-crosextra-carlito fonts-crosextra-caladea fonts-liberation
```

| Microsoft Font | Replacement | Package |
|---|---|---|
| Calibri | Carlito | font-carlito / fonts-crosextra-carlito |
| Cambria | Caladea | font-caladea / fonts-crosextra-caladea |
| Arial, Times New Roman, Courier | Liberation Sans/Serif/Mono | font-liberation / fonts-liberation |

> **TODO:** Build a setup script that automates Python venv creation, pip install, font installation, .env template, and database initialization.

## Commands

```bash
# All commands run from document_pipeline/ using the project venv

# Run all tests with coverage
.venv/bin/python -m pytest --cov=src --cov-report=term-missing

# Run a specific test file
.venv/bin/python -m pytest tests/test_<module>.py -v

# Code quality (all must pass clean)
.venv/bin/python -m black --check src/ tests/
.venv/bin/python -m flake8 src/ tests/
.venv/bin/python -m pylint src/ tests/
```

## Code Conventions

### Documentation Style

**Main functions** (pipeline stages, public API, orchestrator functions):
```python
def classify_page(page_image: bytes, file_type: str) -> PageClassification:
    """Classify page content into Text, Table, or Visual.

    Params:
        page_image: Raw image bytes of the rendered page
        file_type: Source format ("pdf", "docx", "pptx", "xlsx", "csv", "md")

    Returns:
        PageClassification with content_types list and confidence scores

    Example:
        >>> result = classify_page(img_bytes, "pdf")
        >>> result.content_types
        ["text", "table"]
    """
```

**Smaller/helper functions** — concise single-line docstring with params and return:
```python
def extract_text_from_cell(cell: Cell) -> str:
    """Extract cleaned text from a table cell. Params: cell (Cell). Returns: str."""
```

### General Rules

- Minimal comments — code should be self-explanatory. Only comment non-obvious logic
- Function names: snake_case, verb-first, consistent with existing patterns
- Type hints on all function signatures
- No classes unless there's a clear reason — prefer functions and dataclasses/TypedDicts for data
- LLM calls must go through `LLMClient` in `utils/llm.py`, never call OpenAI directly from pipeline code
- All LLM interactions use tool calling (structured output), not freeform text

### Testing & Code Quality

- Every module gets a corresponding test file
- Mock LLM calls in unit tests; use fixtures for database interactions
- Test edge cases: empty documents, single-page docs, tables with no headers, mixed content pages
- **After any code change**, run the full quality gate from `document_pipeline/`:
  1. `.venv/bin/python -m pytest tests/` — all tests must pass
  2. `.venv/bin/python -m black --check src/ tests/` — must pass with no reformatting needed
  3. `.venv/bin/python -m flake8 src/ tests/` — must pass with zero warnings
  4. `.venv/bin/python -m pylint src/ tests/` — must score 10.00/10
- **No suppressions** — do not use `# noqa`, `# pylint: disable`, `# type: ignore`, or any other mechanism to skip checks. Fix the code to satisfy the linter, not the other way around
- **Approved exceptions** — in rare cases where a tool limitation makes 100% impossible (e.g., coverage can't track cross-process execution), a suppression comment may be added only with explicit user approval. Document the reason inline

### Traceability & Logging

Two layers — **debug traces** for drilling into problems, **console logs** for monitoring runs:

**Debug traces** — Per-file, per-stage structured output capturing the full processing detail (inputs, decisions, outputs, errors). Written to a trace store so any file's journey through the pipeline can be inspected after the fact. These are verbose by design.

**Console logging** — Minimal and clean. No per-page spam. Each pipeline stage produces a short **stage summary** at the end showing counts, timings, and any failures. The goal is a scannable run log, not a wall of text.

Pattern for every pipeline stage:
1. Collect per-file debug trace data as the stage runs
2. Log a single summary block when the stage completes
3. Surface errors/warnings inline but keep info-level output to summaries only

### Project Structure

```
document_pipeline/src/ingestion/
├── processors/           # Self-contained per-filetype packages
│   ├── pdf/              #   processor.py, content_chunker.py, prompts/
│   ├── docx/             #   (each processor is independently swappable)
│   ├── pptx/
│   └── xlsx/             #   Also has layout.py, table_eda.py, dense_table.py,
│                         #   content_preparation.py, types.py
├── stages/               # Pipeline orchestration (one module per stage)
│   ├── startup.py        #   Stage 0: connections, locks, logging
│   ├── discovery.py      #   Stage 1: filesystem scan
│   ├── extraction.py     #   Stage 2: routes files to processors
│   ├── content_preparation.py  # Stage 3: thin dispatcher, delegates XLSX to xlsx package
│   └── enrichment.py     #   Stage 4: LLM enrichment per page
├── utils/                # Shared infrastructure only
│   ├── config.py         #   Environment variable access
│   ├── file_types.py     #   All pipeline stage I/O dataclasses
│   ├── llm.py            #   LLM client (swappable auth)
│   ├── postgres.py       #   Database operations
│   ├── oauth.py          #   OAuth token management
│   ├── prompt_loader.py  #   YAML prompt loading
│   ├── logging_setup.py  #   Stage logger setup
│   └── ssl_certificates.py
└── main.py               # Pipeline entry point
```

### Project Structure Patterns

- **Self-contained processors** — each processor package has its own code, prompts, and utils. No shared modules between processors. Copy and modify, never extract shared processing logic
- Pipeline stages are separate modules that delegate filetype-specific work to processor packages
- Configuration via environment variables (never hardcode connection strings, API keys, etc.)
- Database operations isolated in `utils/postgres.py` — pipeline code never writes raw SQL
- All pipeline data flows as JSON files between stages: `processing/discovery.json` → `processing/extraction/` → `processing/content_preparation/` → `processing/enrichment/`
