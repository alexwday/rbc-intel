# Persistent Processor Upgrade Plan

## Summary
This plan is the canonical roadmap for processor upgrades and should be persisted in two places once Plan Mode ends:

- Repo copy: `/Users/alexwday/Projects/rbc-intel/docs/plans/processor-upgrade-plan.md`
- Codex memory copy: `/Users/alexwday/.codex/memories/rbc-intel-processor-upgrade.md`

The plan assumes:
- Processors remain standalone and intentionally duplicated where useful.
- PDF, DOCX, and PPT remain vision-first.
- XLSX remains structure-aware, with vision added for charts/images/dashboard regions.
- Fail-fast means retries plus the approved vision fallback ladder; if a page/slide/sheet is still unrecoverable, the whole file fails.
- Dense-table replacement is a first-pass retrieval router, with second-stage subretrieval over preserved dense-table data.

## Implementation Changes

### Phase 0: Persist the plan
- Create the repo plan document at the path above and store this full plan verbatim.
- Create the Codex memory note at the path above and store the same content with a short header noting it is the canonical processor roadmap.
- Treat the repo copy as team-visible source of truth and the memory copy as continuity backup for context resets.

### Phase 1: Vision foundation and fail-fast contract
- Fix the current vision import/runtime issue so PDF/DOCX/PPT tests collect and run reliably.
- Keep the existing vision fallback ladder intact.
- Remove partial-success behavior from PDF, DOCX, and PPT processors.
- Update extraction orchestration so any unrecovered page/slide/sheet failure fails the file and prevents a successful extraction artifact from being written.
- Keep the top-level extraction contract stable while allowing richer processor-specific metadata.

### Phase 2: PDF processor upgrades
- Add previous-page context to PDF so multi-page tables, lists, and footnotes continue coherently.
- Introduce a PDF-specific vision prompt focused on repeated headers/footers, page furniture, continued tables, continued prose, chart-heavy pages, and footnote marker normalization.
- Add PDF page metadata fields for continuation and repeated-furniture detection.
- Preserve the vision-first design and the shared fallback ladder; do not add non-vision escape paths.

### Phase 3: DOCX processor upgrades
- Keep DOCX vision-first and expand its current page-continuity behavior.
- Introduce a DOCX-specific prompt focused on multi-page tables, section carryover, repeated headers/footers, captions, and legal footer text.
- Add DOCX metadata fields for section continuation, repeated furniture, and table continuation.
- Maintain strict file-level failure semantics.

### Phase 4: PPTX processor upgrades
- Keep the render-to-PDF vision path and add a PPTX-specific prompt tuned for slide semantics.
- Encode slide reading order explicitly: title/subtitle, primary visual or table, callouts, footer/source text.
- Add targeted handling for title slides, agenda slides, chart slides, dashboard slides, comparison slides, and appendix/data slides.
- Add PPTX metadata fields such as slide type guess, chart presence, dashboard presence, and dense-visual-content indicators.

### Phase 5: XLSX processor upgrades
- Keep one sheet-level result per worksheet, but extend sheet metadata to support multiple first-class regions instead of one selected dense region.
- Region types should include framing, dense_table, small_table, visual, and mixed.
- Preserve multiple dense tables per sheet as independent addressable regions.
- Add vision-based extraction for chartsheets, embedded charts, images, and dashboard regions using prompting aligned with the PDF/DOCX/PPT visual extraction style.
- Replace the current count/title-only visual summary with richer visual descriptions while retaining structural metadata such as chart/image counts and native chart titles when available.

### Phase 6: Dense-table replacement and subretrieval handoff
- Keep dense-table replacement for first-pass similarity search.
- Strengthen the replacement payload so it acts as deterministic routing instructions for second-stage table subretrieval.
- Every replaced dense-table chunk must carry stable routing metadata: workbook/sheet identity, region ID, used range, source region ID, and query-routing column roles.
- Preserve the underlying dense-table data and region structure as the system of record for second-stage answering.

## Public Interfaces and Metadata Additions
- `ExtractionResult` and `PageResult` stay as the common processor contract.
- PDF/DOCX/PPT metadata additions should cover continuation state and repeated page furniture.
- PPTX metadata should also cover slide type and visual density signals.
- XLSX metadata should evolve from single dense-table fields to a multi-region schema while retaining backward-compatible top-level sheet classification where needed.
- Dense-table replacement chunks should expose deterministic routing fields needed by the second-stage table retriever.

## Test Plan
- Add fail-fast tests proving that one unrecovered page/slide/sheet causes full-file failure.
- Add processor-specific golden cases:
  - PDF: multi-page tables, repeated headers/footers, chart-heavy pages.
  - DOCX: multi-page tables, repeated page furniture, section continuation.
  - PPTX: chart slides, dashboard slides, comparison layouts.
  - XLSX: mixed-content sheets, multiple dense tables on one sheet, chartsheets, visual-only dashboard tabs, formula/cached-value cases.
- Add regression tests for the dense-table replacement handoff so a first-pass retrieval hit routes deterministically to the correct dense-table region.
- Add a benchmark suite comparing current output quality against a baseline OSS XLSX partitioner behavior for mixed sheets and multi-table sheets.

## Assumptions and Defaults
- Default persistence target is both repo and Codex memory.
- No processor code-sharing refactor is planned beyond the already-accepted shared vision ladder.
- No native-structure-first redesign is planned for PDF/DOCX/PPT.
- XLSX is the only processor that should substantially diverge from the page-like model because of dense-table and mixed-sheet requirements.
- If a future context reset happens before persistence is written, this plan block is the temporary canonical copy.
